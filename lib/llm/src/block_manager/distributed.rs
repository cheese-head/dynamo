// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod remote;
pub mod transfer;
mod utils;
pub mod vllm;
mod zmq;

mod leader;
pub mod notifications;
pub mod registry;
mod worker;

pub use notifications::{
    CompletionChecker, NixlNotificationSender, NixlStatusChecker, RegisterTransferNotification,
    TransferCompleteNotification, spawn_notification_handler,
};
pub use remote::{
    CanOffloadResult, PositionalRemoteHandle, RemoteHandle, RemoteHashOperations,
    RemoteHashOperationsSync, RemoteOperation,
};

pub use leader::{G4InflightTracker, KvbmLeader, KvbmLeaderConfig, KvbmLeaderNumBlocksConfig};
pub use transfer::BlockTransferHandler;
pub use utils::{
    BlockTransferPool, BlockTransferRequest, ConnectorRequestLeader, ConnectorTransferType,
    RemoteTransferRequest, SerializableRemoteBlockDescriptor, SerializableRemoteTransferPipeline,
    SerializableStorageType, SerializableTransferDirection, ZMQ_REMOTE_TRANSFER_MESSAGE,
};
pub use worker::{KvbmWorker, KvbmWorkerConfig};
pub use zmq::Handler;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskReady {
    Continue,
    Cancel,
}
#[async_trait::async_trait]
pub trait ScheduledTaskHandle: Send + Sync {
    async fn ready(&self) -> TaskReady;
    fn mark_complete(&self);
}

pub struct SchedulerRequest<T> {
    pub handle_tx: tokio::sync::oneshot::Sender<Box<dyn ScheduledTaskHandle>>,
    pub task: T,
}

// impl<T> SchedulerRequest<T> {
//     pub fn new(task: T) -> (Self, tokio::sync::oneshot::Sender<Box<dyn ScheduledTaskHandle>>) {
//         let (handle_tx, handle_rx) = tokio::sync::oneshot::channel();
//         Self { handle_tx, task }
//     }
// }

/// Leader + worker + `KvBlockManager` NIXL integration tests (real CUDA device tensors, POSIX disk).
///
/// Run (example): `cargo test -p dynamo-llm e2e_harness --features testing-cuda -- --nocapture`
#[cfg(all(test, feature = "testing-cuda"))]
pub(crate) mod tests {
    use super::*;

    use crate::block_manager::KvBlockManager;
    use crate::block_manager::block::BasicMetadata;
    use crate::block_manager::block::data::logical::distributed_leader_worker::DistributedLeaderWorkerResources;
    use crate::block_manager::config::*;
    use crate::block_manager::locality::Logical;
    use crate::block_manager::storage::{
        DeviceAllocator, Storage, StorageAllocator,
        torch::{TorchDevice, TorchTensor},
    };

    use anyhow::Result;
    use rstest::*;
    use serial_test::serial;

    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };
    use std::sync::Once;
    use tokio_util::sync::CancellationToken;

    use dynamo_runtime::logging::init as init_logging;

    const DEFAULT_NUM_BLOCKS: usize = 8;
    const TEST_NUM_LAYERS: usize = 2;
    const TEST_OUTER_DIM: usize = 2;
    const TEST_INNER_DIM: usize = 4;
    static TEST_PORT_COUNTER: AtomicUsize = AtomicUsize::new(0);
    static TEST_ENV_INIT: Once = Once::new();

    #[derive(Clone, Debug)]
    struct MockTensor {
        ptr: u64,
        size: usize,
        shape: Vec<usize>,
    }

    impl MockTensor {
        fn new(shape: Vec<usize>) -> Self {
            let allocator = DeviceAllocator::new(0).unwrap();

            // Multiply by 2 for fp16.
            let size = shape.iter().product::<usize>() * 2;

            let device_storage = std::mem::ManuallyDrop::new(allocator.allocate(size).unwrap());

            let ptr = device_storage.addr();
            Self { ptr, size, shape }
        }
    }

    impl TorchTensor for MockTensor {
        fn device(&self) -> TorchDevice {
            TorchDevice::Cuda(0)
        }

        fn data_ptr(&self) -> u64 {
            self.ptr
        }

        fn size_bytes(&self) -> usize {
            self.size
        }

        fn shape(&self) -> Vec<usize> {
            self.shape.clone()
        }

        fn stride(&self) -> Vec<usize> {
            // Generate the stride on the assumption that it is contiguous.
            let mut stride = vec![1];
            for i in (0..self.shape.len() - 1).rev() {
                stride.push(stride.last().unwrap() * self.shape[i]);
            }
            stride.reverse();
            stride
        }
    }

    fn next_test_ports() -> (String, String) {
        let offset = TEST_PORT_COUNTER.fetch_add(1, Ordering::Relaxed);
        let base_port = 56001 + (offset * 2);
        (
            format!("tcp://127.0.0.1:{base_port}"),
            format!("tcp://127.0.0.1:{}", base_port + 1),
        )
    }

    fn init_test_env() {
        TEST_ENV_INIT.call_once(|| {
            // The default NIXL POSIX queue selection may pick linux_aio in this container,
            // which is not stable enough for the distributed E2E suite. Force a deterministic
            // backend for tests so disk offload/onboard validates the KVBM path instead of the
            // host's AIO limits.
            unsafe {
                std::env::set_var("DYN_KVBM_NIXL_POSIX_API", "posix_aio");
                std::env::set_var("DYN_KVBM_LOCAL_DISK_USE_GDS", "false");
            }
        });
    }

    pub(crate) async fn build_leader_and_workers(
        num_workers: usize,
        num_device_blocks: usize,
        num_host_blocks: usize,
        num_disk_blocks: usize,
        page_size: usize,
    ) -> Result<(KvbmLeader, Vec<KvbmWorker>)> {
        init_test_env();
        let (leader_pub_url, leader_ack_url) = next_test_ports();
        let mut workers = Vec::new();

        for _ in 0..num_workers {
            let tensors: Vec<Arc<dyn TorchTensor>> = vec![Arc::new(MockTensor::new(vec![
                num_device_blocks,
                TEST_NUM_LAYERS,
                TEST_OUTER_DIM,
                page_size * TEST_INNER_DIM,
            ]))];

            let config = KvbmWorkerConfig::builder()
                .cancel_token(CancellationToken::new())
                .num_device_blocks(num_device_blocks)
                .page_size(page_size)
                .tensors(tensors)
                .device_id(0)
                .leader_pub_url(leader_pub_url.clone())
                .leader_ack_url(leader_ack_url.clone())
                .build()?;

            let worker = KvbmWorker::new(config, false).await?;
            workers.push(worker);
        }

        let host_blocks = KvbmLeaderNumBlocksConfig {
            cache_size_in_gb: 1.0,
            num_blocks_overriden: num_host_blocks,
        };

        let disk_blocks = KvbmLeaderNumBlocksConfig {
            cache_size_in_gb: 1.0,
            num_blocks_overriden: num_disk_blocks,
        };

        let leader_config = KvbmLeaderConfig::builder()
            .world_size(num_workers)
            .host_blocks_config(host_blocks)
            .disk_blocks_config(disk_blocks)
            .leader_pub_url(leader_pub_url)
            .leader_ack_url(leader_ack_url)
            .build()?;

        // When/if this returns, we know that all the workers were also successful.
        let leader = KvbmLeader::new(leader_config).await?;
        anyhow::ensure!(
            leader.wait_worker_sync_ready().await,
            "timed out waiting for leader/worker ZMQ handshake readiness"
        );

        Ok((leader, workers))
    }

    pub(crate) async fn build_test_block_manager(
        leader: Arc<KvbmLeader>,
        num_blocks: usize,
        block_size: usize,
    ) -> Result<KvBlockManager<Logical<DistributedLeaderWorkerResources>, BasicMetadata>> {
        let cancel_token = CancellationToken::new();

        let config = KvBlockManagerConfig::builder()
            .runtime(
                KvManagerRuntimeConfig::builder()
                    .worker_id(0)
                    .cancellation_token(cancel_token.clone())
                    .build()?,
            )
            .model(
                KvManagerModelConfig::builder()
                    .num_layers(TEST_NUM_LAYERS)
                    .outer_dim(TEST_OUTER_DIM)
                    .page_size(block_size)
                    .inner_dim(TEST_INNER_DIM)
                    .build()?,
            )
            .device_layout(
                KvManagerLayoutConfig::builder()
                    .num_blocks(num_blocks)
                    .logical(Some(BlockParallelismStrategy::LeaderWorkerSharded))
                    .build()?,
            )
            .host_layout(
                KvManagerLayoutConfig::builder()
                    .num_blocks(num_blocks)
                    .logical(Some(BlockParallelismStrategy::LeaderWorkerSharded))
                    .build()?,
            )
            .disk_layout(
                KvManagerLayoutConfig::builder()
                    .num_blocks(num_blocks)
                    .logical(Some(BlockParallelismStrategy::LeaderWorkerSharded))
                    .build()?,
            )
            .build()?;

        let resources =
            DistributedLeaderWorkerResources::new(Some(leader), cancel_token.child_token())?;

        Ok(
            KvBlockManager::<Logical<DistributedLeaderWorkerResources>, BasicMetadata>::new(
                config, resources,
            )
            .await
            ?,
        )
    }

    #[tokio::test]
    #[rstest]
    #[serial]
    #[case(1)]
    #[case(2)]
    #[case(4)]
    #[case(8)]
    async fn test_leader_worker_sync_and_transfer(#[case] num_workers: usize) -> Result<()> {
        init_logging();

        let (leader, _workers) = build_leader_and_workers(
            num_workers,
            DEFAULT_NUM_BLOCKS,
            DEFAULT_NUM_BLOCKS,
            DEFAULT_NUM_BLOCKS,
            32,
        )
        .await?;

        // Do a whole bunch of distributed transfers.

        for block_idx in 0..DEFAULT_NUM_BLOCKS {
            leader
                .transfer_blocks_request(utils::BlockTransferRequest::new(
                    utils::BlockTransferPool::Device,
                    utils::BlockTransferPool::Host,
                    vec![(block_idx, block_idx)],
                ))
                .await?
                .await?;
        }

        for block_idx in 0..DEFAULT_NUM_BLOCKS {
            leader
                .transfer_blocks_request(utils::BlockTransferRequest::new(
                    utils::BlockTransferPool::Host,
                    utils::BlockTransferPool::Disk,
                    vec![(block_idx, block_idx)],
                ))
                .await?
                .await?;
        }

        Ok(())
    }

    #[tokio::test]
    #[rstest]
    #[serial]
    #[case(1, 8, 4, 100)]
    #[case(2, 8, 4, 100)]
    #[case(4, 8, 8, 150)]
    async fn test_leader_worker_transfer_e2e(
        #[case] num_workers: usize,
        #[case] num_blocks: usize,
        #[case] block_size: usize,
        #[case] wait_ms: u64,
    ) -> Result<()> {
        init_logging();

        let (leader, _workers) =
            build_leader_and_workers(num_workers, num_blocks, num_blocks, num_blocks, block_size)
                .await?;

        let block_manager =
            build_test_block_manager(Arc::new(leader), num_blocks, block_size).await?;

        let device_pool = block_manager.device().unwrap();
        let host_pool = block_manager.host().unwrap();
        let disk_pool = block_manager.disk().unwrap();

        let mut device_blocks = device_pool.allocate_blocks(num_blocks).await?;

        let mut sequence_hashes = Vec::new();
        for block in &mut device_blocks {
            block.init_sequence(42).unwrap();

            for _ in 0..block_size {
                block.add_token(42).unwrap();
            }

            block.commit().unwrap();

            sequence_hashes.push(block.sequence_hash().unwrap());
        }

        // Register our blocks on the device.
        let immutable_device_blocks = device_pool.register_blocks(device_blocks).await?;

        // Wait for the blocks to be offloaded.
        tokio::time::sleep(std::time::Duration::from_millis(wait_ms)).await;

        // Now, all blocks should be on the host.
        let host_blocks = host_pool
            .match_sequence_hashes(sequence_hashes.as_slice())
            .await?;

        assert_eq!(host_blocks.len(), num_blocks);

        let disk_blocks = disk_pool
            .match_sequence_hashes(sequence_hashes.as_slice())
            .await?;

        assert_eq!(disk_blocks.len(), num_blocks);

        // Return the device blocks to the pool.
        drop(immutable_device_blocks);

        tokio::time::sleep(std::time::Duration::from_millis(wait_ms)).await;

        // Clear out the device pool.
        let _ = device_pool.allocate_blocks(num_blocks).await?;

        // Now, all the blocks should be gone.
        assert_eq!(
            device_pool
                .match_sequence_hashes(sequence_hashes.as_slice())
                .await?
                .len(),
            0
        );

        // Wait for the device blocks to be returned to the pool.
        tokio::time::sleep(std::time::Duration::from_millis(wait_ms)).await;

        // Now, onboard them back to the device.
        let new_device_blocks = block_manager.onboard_blocks(host_blocks, None).await??;

        assert_eq!(new_device_blocks.len(), num_blocks);

        Ok(())
    }

    #[tokio::test]
    #[rstest]
    #[serial]
    #[case(1, 8, 4, 150)]
    #[case(2, 8, 4, 150)]
    #[case(4, 8, 8, 200)]
    async fn test_leader_worker_disk_onboard_e2e(
        #[case] num_workers: usize,
        #[case] num_blocks: usize,
        #[case] block_size: usize,
        #[case] wait_ms: u64,
    ) -> Result<()> {
        init_logging();

        let (leader, _workers) =
            build_leader_and_workers(num_workers, num_blocks, num_blocks, num_blocks, block_size)
                .await?;
        let block_manager =
            build_test_block_manager(Arc::new(leader), num_blocks, block_size).await?;

        let device_pool = block_manager.device().unwrap();
        let host_pool = block_manager.host().unwrap();
        let disk_pool = block_manager.disk().unwrap();

        let mut device_blocks = device_pool.allocate_blocks(num_blocks).await?;
        let mut sequence_hashes = Vec::new();

        for (idx, block) in device_blocks.iter_mut().enumerate() {
            block.init_sequence(100 + idx as u64).unwrap();
            for token in 0..block_size {
                block.add_token((idx * block_size + token) as u32).unwrap();
            }
            let metadata = block.metadata().update_priority((idx as u32) + 1);
            block.update_metadata(metadata);
            block.commit().unwrap();
            sequence_hashes.push(block.sequence_hash().unwrap());
        }

        let immutable_device_blocks = device_pool.register_blocks(device_blocks).await?;

        tokio::time::sleep(std::time::Duration::from_millis(wait_ms)).await;

        let host_blocks = host_pool
            .match_sequence_hashes(sequence_hashes.as_slice())
            .await?;
        assert_eq!(host_blocks.len(), num_blocks);

        let disk_blocks = disk_pool
            .match_sequence_hashes(sequence_hashes.as_slice())
            .await?;
        assert_eq!(disk_blocks.len(), num_blocks);

        for (expected_idx, disk_block) in disk_blocks.iter().enumerate() {
            assert_eq!(disk_block.metadata().priority(), (expected_idx as u32) + 1);
        }

        drop(immutable_device_blocks);
        tokio::time::sleep(std::time::Duration::from_millis(wait_ms)).await;
        let _ = device_pool.allocate_blocks(num_blocks).await?;
        tokio::time::sleep(std::time::Duration::from_millis(wait_ms)).await;

        let onboarded_from_disk = block_manager.onboard_blocks(disk_blocks, None).await??;
        assert_eq!(onboarded_from_disk.len(), num_blocks);
        for (expected_idx, device_block) in onboarded_from_disk.iter().enumerate() {
            assert_eq!(
                device_block.metadata().priority(),
                (expected_idx as u32) + 1
            );
        }

        Ok(())
    }

    #[tokio::test]
    #[rstest]
    #[serial]
    #[case(1, 8, 4, 4, 8, 150)]
    #[case(2, 8, 4, 6, 8, 150)]
    async fn test_leader_worker_transfer_rejects_when_host_capacity_too_small(
        #[case] num_workers: usize,
        #[case] num_blocks: usize,
        #[case] block_size: usize,
        #[case] num_host_blocks: usize,
        #[case] num_disk_blocks: usize,
        #[case] wait_ms: u64,
    ) -> Result<()> {
        init_logging();

        let (leader, _workers) = build_leader_and_workers(
            num_workers,
            num_blocks,
            num_host_blocks,
            num_disk_blocks,
            block_size,
        )
        .await?;
        let block_manager =
            build_test_block_manager(Arc::new(leader), num_blocks, block_size).await?;

        let device_pool = block_manager.device().unwrap();
        let host_pool = block_manager.host().unwrap();
        let disk_pool = block_manager.disk().unwrap();

        let mut device_blocks = device_pool.allocate_blocks(num_blocks).await?;
        let mut sequence_hashes = Vec::new();

        for (idx, block) in device_blocks.iter_mut().enumerate() {
            block.init_sequence(7 + idx as u64).unwrap();
            for token in 0..block_size {
                block.add_token((idx * block_size + token) as u32).unwrap();
            }
            block.commit().unwrap();
            sequence_hashes.push(block.sequence_hash().unwrap());
        }

        let _immutable_device_blocks = device_pool.register_blocks(device_blocks).await?;

        tokio::time::sleep(std::time::Duration::from_millis(wait_ms)).await;

        let host_blocks = host_pool
            .match_sequence_hashes(sequence_hashes.as_slice())
            .await?;
        let disk_blocks = disk_pool
            .match_sequence_hashes(sequence_hashes.as_slice())
            .await?;

        assert!(
            host_blocks.len() < num_blocks,
            "host pool unexpectedly retained all blocks despite constrained capacity"
        );
        assert_eq!(
            disk_blocks.len(),
            0,
            "disk persistence should not occur when host admission fails on the G1->G2->G3 path"
        );

        Ok(())
    }

    // -------------------------------------------------------------------------
    // End-to-end harness: N repeated “requests” (cycles) through full tier motion.
    // Each cycle: device blocks (unique prefix) → background offload to host + disk,
    // then onboard back to device (from host or from disk). Uses the same leader/worker
    // and block manager as production-style distributed KVBM (not the vLLM connector slot).
    // -------------------------------------------------------------------------

    /// For each cycle: fill device → register → wait → assert host+disk hits → release device
    /// reference → onboard from **host** → drop onboarded GPU blocks (returns capacity).
    #[rstest]
    #[case(1)]
    #[case(3)]
    #[tokio::test]
    #[serial]
    async fn e2e_harness_n_cycles_offload_host_disk_then_onboard_from_host(
        #[case] num_cycles: usize,
    ) -> Result<()> {
        init_logging();

        let num_blocks = 8usize;
        let block_size = 4usize;
        let wait_ms = 150u64;

        let (leader, _workers) = build_leader_and_workers(
            1,
            num_blocks,
            num_blocks,
            num_blocks,
            block_size,
        )
        .await?;
        let leader = Arc::new(leader);
        let block_manager =
            build_test_block_manager(leader.clone(), num_blocks, block_size).await?;

        let device_pool = block_manager.device().unwrap();
        let host_pool = block_manager.host().unwrap();
        let disk_pool = block_manager.disk().unwrap();

        for cycle in 0..num_cycles {
            let salt = 10_000u64 + cycle as u64;
            let mut device_blocks = device_pool.allocate_blocks(num_blocks).await?;
            let mut sequence_hashes = Vec::new();

            for (idx, block) in device_blocks.iter_mut().enumerate() {
                block.init_sequence(salt + idx as u64).unwrap();
                for token in 0..block_size {
                    let t = (cycle * 100_000 + idx * block_size + token) as u32;
                    block.add_token(t).unwrap();
                }
                block.commit().unwrap();
                sequence_hashes.push(block.sequence_hash().unwrap());
            }

            let immutable_device_blocks = device_pool.register_blocks(device_blocks).await?;
            tokio::time::sleep(std::time::Duration::from_millis(wait_ms)).await;

            let host_blocks = host_pool
                .match_sequence_hashes(sequence_hashes.as_slice())
                .await?;
            assert_eq!(
                host_blocks.len(),
                num_blocks,
                "cycle {cycle}: expected full host prefix"
            );
            let disk_blocks = disk_pool
                .match_sequence_hashes(sequence_hashes.as_slice())
                .await?;
            assert_eq!(
                disk_blocks.len(),
                num_blocks,
                "cycle {cycle}: expected full disk prefix"
            );

            drop(immutable_device_blocks);
            tokio::time::sleep(std::time::Duration::from_millis(wait_ms)).await;
            let _ = device_pool.allocate_blocks(num_blocks).await?;
            tokio::time::sleep(std::time::Duration::from_millis(wait_ms)).await;

            assert_eq!(
                device_pool
                    .match_sequence_hashes(sequence_hashes.as_slice())
                    .await?
                    .len(),
                0,
                "cycle {cycle}: device pool should not retain committed sequence"
            );

            let onboarded = block_manager.onboard_blocks(host_blocks, None).await??;
            assert_eq!(
                onboarded.len(),
                num_blocks,
                "cycle {cycle}: host→device onboard"
            );
            drop(onboarded);
            tokio::time::sleep(std::time::Duration::from_millis(wait_ms)).await;
        }

        Ok(())
    }

    /// Same offload path; onboard from **disk** (priorities preserved like `test_leader_worker_disk_onboard_e2e`).
    #[rstest]
    #[case(1)]
    #[case(2)]
    #[tokio::test]
    #[serial]
    async fn e2e_harness_n_cycles_offload_host_disk_then_onboard_from_disk(
        #[case] num_cycles: usize,
    ) -> Result<()> {
        init_logging();

        let num_blocks = 8usize;
        let block_size = 4usize;
        let wait_ms = 150u64;

        let (leader, _workers) = build_leader_and_workers(
            1,
            num_blocks,
            num_blocks,
            num_blocks,
            block_size,
        )
        .await?;
        let leader = Arc::new(leader);
        let block_manager =
            build_test_block_manager(leader.clone(), num_blocks, block_size).await?;

        let device_pool = block_manager.device().unwrap();
        let host_pool = block_manager.host().unwrap();
        let disk_pool = block_manager.disk().unwrap();

        for cycle in 0..num_cycles {
            let salt = 50_000u64 + cycle as u64;
            let mut device_blocks = device_pool.allocate_blocks(num_blocks).await?;
            let mut sequence_hashes = Vec::new();

            for (idx, block) in device_blocks.iter_mut().enumerate() {
                block.init_sequence(salt + idx as u64).unwrap();
                for token in 0..block_size {
                    let t = (cycle * 10_000 + idx * block_size + token) as u32;
                    block.add_token(t).unwrap();
                }
                let metadata = block.metadata().update_priority((idx as u32) + 1);
                block.update_metadata(metadata);
                block.commit().unwrap();
                sequence_hashes.push(block.sequence_hash().unwrap());
            }

            let immutable_device_blocks = device_pool.register_blocks(device_blocks).await?;
            tokio::time::sleep(std::time::Duration::from_millis(wait_ms)).await;

            let host_blocks = host_pool
                .match_sequence_hashes(sequence_hashes.as_slice())
                .await?;
            assert_eq!(host_blocks.len(), num_blocks, "cycle {cycle}: host");
            let disk_blocks = disk_pool
                .match_sequence_hashes(sequence_hashes.as_slice())
                .await?;
            assert_eq!(disk_blocks.len(), num_blocks, "cycle {cycle}: disk");

            for (expected_idx, disk_block) in disk_blocks.iter().enumerate() {
                assert_eq!(disk_block.metadata().priority(), (expected_idx as u32) + 1);
            }

            drop(immutable_device_blocks);
            tokio::time::sleep(std::time::Duration::from_millis(wait_ms)).await;
            let _ = device_pool.allocate_blocks(num_blocks).await?;
            tokio::time::sleep(std::time::Duration::from_millis(wait_ms)).await;

            let from_disk = block_manager.onboard_blocks(disk_blocks, None).await??;
            assert_eq!(from_disk.len(), num_blocks, "cycle {cycle}: disk→device");
            for (expected_idx, device_block) in from_disk.iter().enumerate() {
                assert_eq!(
                    device_block.metadata().priority(),
                    (expected_idx as u32) + 1
                );
            }
            drop(from_disk);
            tokio::time::sleep(std::time::Duration::from_millis(wait_ms)).await;
        }

        Ok(())
    }
}
