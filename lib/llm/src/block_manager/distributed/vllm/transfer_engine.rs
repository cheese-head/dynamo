// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{sync::Arc, time::Instant};

use dynamo_runtime::utils::task::CriticalTaskExecutionHandle;
use tokio::task::JoinSet;
use tokio::{runtime::Handle, sync::mpsc};
use tokio_util::sync::CancellationToken;
use tracing::Instrument;

use crate::block_manager::KvBlockManager;
use crate::block_manager::{
    BasicMetadata, BlockMetadata, BlockPool, NixlRegisterableStorage, Storage,
    block::{
        data::logical::distributed_leader_worker::DistributedLeaderWorkerResources,
        locality::{LocalityProvider, Logical},
    },
    config::{RemoteStorageConfig, should_bypass_cpu_cache},
    connector::protocol::{LeaderTransferRequest, RequestType},
    distributed::{BlockTransferPool, BlockTransferRequest, KvbmLeader, vllm as vllm_int},
    metrics_kvbm::KvbmMetrics,
    pool::PinRegistry,
    transfer_orchestrator::{TransferPriority, priority_channel, run_priority_worker},
};

use super::{
    DrainItem, LocalOffloadRequest, LocalOnboardRequest, LocalTransferRequest,
    RemoteTransferRequest,
};

type VllmBlockManager = KvBlockManager<Logical<DistributedLeaderWorkerResources>, BasicMetadata>;

use super::g4_transfer_timeout;

pub struct LocalTransferEngine {
    block_manager: VllmBlockManager,
    leader: Arc<KvbmLeader>,
    xfer_rx: mpsc::UnboundedReceiver<LocalTransferRequest>,
}

impl LocalTransferEngine {
    pub fn new(
        block_manager: VllmBlockManager,
        leader: Arc<KvbmLeader>,
        xfer_rx: mpsc::UnboundedReceiver<LocalTransferRequest>,
    ) -> Self {
        Self {
            block_manager,
            leader,
            xfer_rx,
        }
    }

    pub async fn execute(
        &mut self,
        cancellation_token: CancellationToken,
        task_handle: Handle,
        task_token: CancellationToken,
        kvbm_metrics: KvbmMetrics,
        pin_registry: PinRegistry,
        prefetch_completed: Arc<std::sync::Mutex<std::collections::HashSet<uuid::Uuid>>>,
        prefetch_failed: Arc<std::sync::Mutex<std::collections::HashSet<uuid::Uuid>>>,
        signal: Arc<dyn super::transfer_signal::TransferSignal>,
    ) -> anyhow::Result<()> {
        let (onboard_tx, mut onboard_rx) = mpsc::unbounded_channel::<LocalOnboardRequest>();
        let (offload_tx, mut offload_rx) = mpsc::unbounded_channel::<LocalOffloadRequest>();
        let (remote_tx, remote_rx) = priority_channel::<RemoteTransferRequest>(0, 0);
        let (drain_tx, mut drain_rx) = mpsc::unbounded_channel::<DrainItem>();

        let pin_registry_drain = pin_registry.clone();
        let pin_registry_remote = pin_registry.clone();

        let block_manager_offload = self.block_manager.clone();
        let block_manager_remote = self.block_manager.clone();
        let leader_offload = Arc::clone(&self.leader);
        let leader_onboard = Arc::clone(&self.leader);
        let leader_remote = Arc::clone(&self.leader);
        let drain_tx_for_offload = drain_tx.clone();
        let remote_tx_for_drain = remote_tx.clone();

        let kvbm_metrics_onboard = kvbm_metrics.clone();
        let kvbm_metrics_offload = kvbm_metrics.clone();
        let kvbm_metrics_remote = kvbm_metrics.clone();

        let signal_offload = Arc::clone(&signal);
        let signal_remote = Arc::clone(&signal);

        let onboard_task = CriticalTaskExecutionHandle::new_with_runtime(
            |cancellation_token_onboard| async move {
                let mut join_set = JoinSet::new();

                loop {
                    tokio::select! {
                        _ = cancellation_token_onboard.cancelled() => break,
                        done = join_set.join_next(), if !join_set.is_empty() => {
                            if let Some(Err(e)) = done {
                                tracing::error!("LocalOnboardTask join error: {:?}", e);
                            }
                        }
                        req = onboard_rx.recv() => {
                            match req {
                                Some(req) => {
                                    let leader = Arc::clone(&leader_onboard);
                                    let metrics = kvbm_metrics_onboard.clone();
                                    let sig = Arc::clone(&signal);
                                    join_set.spawn(async move {
                                        let op_id = req.operation_id;
                                        match process_onboard_request(req, &leader, metrics).await {
                                            Ok(()) => {
                                                sig.complete(op_id);
                                            }
                                            Err(e) => {
                                                tracing::error!("LocalOnboardTask error: {:?}", e);
                                                sig.fail(op_id);
                                            }
                                        }
                                    });
                                }
                                None => break,
                            }
                        }
                    }
                }

                while let Some(done) = join_set.join_next().await {
                    if let Err(e) = done {
                        tracing::error!("LocalOnboardTask join error: {:?}", e);
                    }
                }
                Ok(())
            },
            task_token.clone(),
            "LocalOnboardTask",
            &task_handle,
        )
        .unwrap();

        let offload_task = CriticalTaskExecutionHandle::new_with_runtime(
            |cancellation_token_offload| async move {
                let mut join_set = JoinSet::new();

                loop {
                    tokio::select! {
                        _ = cancellation_token_offload.cancelled() => break,
                        done = join_set.join_next(), if !join_set.is_empty() => {
                            if let Some(Err(e)) = done {
                                tracing::error!("LocalOffloadTask join error: {:?}", e);
                            }
                        }
                        req = offload_rx.recv() => {
                            match req {
                                Some(req) => {
                                    let key = req.key.clone();
                                    let operation_id = req.operation_id;
                                    let block_manager = block_manager_offload.clone();
                                    let leader = Arc::clone(&leader_offload);
                                    let metrics = kvbm_metrics_offload.clone();
                                    let drain_tx = drain_tx_for_offload.clone();
                                    let sig = Arc::clone(&signal_offload);

                                    join_set.spawn(async move {
                                        match process_offload_request(
                                            req,
                                            &block_manager,
                                            &leader,
                                            metrics,
                                            &drain_tx,
                                        )
                                        .await
                                        {
                                            Ok(()) => {
                                                sig.complete(operation_id);
                                            }
                                            Err(e) => {
                                                tracing::error!("LocalOffloadTask error: {:?}", e);
                                                sig.fail(operation_id);
                                                let fake_xfer = BlockTransferRequest {
                                                    from_pool: BlockTransferPool::Device,
                                                    to_pool: BlockTransferPool::Host,
                                                    blocks: vec![],
                                                    connector_req: Some(LeaderTransferRequest {
                                                        key: key.clone(),
                                                        uuid: operation_id,
                                                        requirement: None,
                                                        request_type: RequestType::Immediate,
                                                        chained: false,
                                                    }),
                                                    sequence_hashes: None,
                                                    traceparent: None,
                                                };
                                                if let Ok(notify_receiver) = leader.transfer_blocks_request(fake_xfer).await {
                                                    let _ = notify_receiver.await;
                                                }
                                            }
                                        }
                                    });
                                }
                                None => break,
                            }
                        }
                    }
                }

                while let Some(done) = join_set.join_next().await {
                    if let Err(e) = done {
                        tracing::error!("LocalOffloadTask join error: {:?}", e);
                    }
                }
                Ok(())
            },
            task_token.clone(),
            "LocalOffloadTask",
            &task_handle,
        )
        .unwrap();

        let prefetch_completed_remote = prefetch_completed;
        let prefetch_failed_remote = prefetch_failed;
        let remote_task = CriticalTaskExecutionHandle::new_with_runtime(
            |cancellation_token_remote| async move {
                run_priority_worker(
                    cancellation_token_remote,
                    remote_rx,
                    0,
                    move |req| {
                        let block_manager = block_manager_remote.clone();
                        let leader = Arc::clone(&leader_remote);
                        let metrics = kvbm_metrics_remote.clone();
                        let pin_reg = pin_registry_remote.clone();
                        let prefetch_ok = prefetch_completed_remote.clone();
                        let prefetch_err = prefetch_failed_remote.clone();
                        let sig = Arc::clone(&signal_remote);
                        async move {
                            let op_id = req.operation_id;
                            let is_prefetch = req.is_onboard
                                && req.device_block_ids.is_empty();
                            match process_remote_transfer_request(
                                req,
                                &block_manager,
                                &leader,
                                metrics,
                                &pin_reg,
                            )
                            .await
                            {
                                Ok(()) if is_prefetch => {
                                    if let Ok(mut set) = prefetch_ok.lock() {
                                        set.insert(op_id);
                                    }
                                }
                                Err(e) if is_prefetch => {
                                    tracing::error!("RemoteTransferTask (prefetch) error: {:?}", e);
                                    if let Ok(mut set) = prefetch_err.lock() {
                                        set.insert(op_id);
                                    }
                                }
                                Ok(()) => {
                                    sig.complete(op_id);
                                }
                                Err(e) => {
                                    tracing::error!("RemoteTransferTask error: {:?}", e);
                                    sig.fail(op_id);
                                }
                            }
                        }
                    },
                )
                .await;
                Ok(())
            },
            task_token.clone(),
            "RemoteTransferTask",
            &task_handle,
        )
        .unwrap();

        let drain_task = CriticalTaskExecutionHandle::new_with_runtime(
            |cancellation_token_drain| async move {
                loop {
                    let item = tokio::select! {
                        _ = cancellation_token_drain.cancelled() => break,
                        item = drain_rx.recv() => match item { Some(item) => item, None => break }
                    };

                    let drain_span = item.traceparent.as_deref()
                        .map(|tp| dynamo_runtime::logging::make_linked_span("kvbm.drain", tp))
                        .unwrap_or_else(|| tracing::info_span!(
                            "drain_item",
                            otel.name = "kvbm.drain",
                            description = "Async host-to-disk drain after offload",
                            request_id = %item.request_id,
                            num_blocks = item.host_block_ids.len(),
                        ));

                    let h2o_operation_id = uuid::Uuid::new_v4();
                    let h2o_req = {
                        let _entered = drain_span.enter();
                        pin_registry_drain.insert(h2o_operation_id, item.pin_guard);

                        RemoteTransferRequest::new_h2o(
                            item.key,
                            item.sequence_hashes,
                            item.host_block_ids,
                            h2o_operation_id,
                            item.block_size,
                            h2o_operation_id,
                            item.traceparent,
                            item.baggage,
                        )
                    };

                    if remote_tx_for_drain
                        .send(TransferPriority::Low, h2o_req)
                        .await
                        .is_err()
                    {
                        pin_registry_drain.remove(&h2o_operation_id);
                        continue;
                    }
                }
                Ok(())
            },
            task_token,
            "DrainTask",
            &task_handle,
        )
        .unwrap();

        loop {
            tokio::select! {
                _ = cancellation_token.cancelled() => break,
                req = self.xfer_rx.recv() => {
                    match req {
                        Some(LocalTransferRequest::Offload(offload_req)) => { let _ = offload_tx.send(offload_req); }
                        Some(LocalTransferRequest::Onboard(onboard_req)) => { let _ = onboard_tx.send(onboard_req); }
                        Some(LocalTransferRequest::Remote(remote_req)) => {
                            if remote_req.is_onboard {
                                let _ = remote_tx.send(TransferPriority::High, remote_req).await;
                            } else {
                                let _ = remote_tx.send(TransferPriority::Low, remote_req).await;
                            }
                        }
                        None => break,
                    }
                }
            }
        }

        drop(onboard_tx);
        drop(offload_tx);
        drop(remote_tx);
        onboard_task.cancel();
        offload_task.cancel();
        remote_task.cancel();
        drain_task.cancel();
        let _ = onboard_task.join().await;
        let _ = offload_task.join().await;
        let _ = remote_task.join().await;
        let _ = drain_task.join().await;
        Ok(())
    }
}

async fn process_offload_request(
    offload_req: LocalOffloadRequest,
    block_manager: &VllmBlockManager,
    leader: &Arc<KvbmLeader>,
    kvbm_metrics: KvbmMetrics,
    drain_tx: &mpsc::UnboundedSender<DrainItem>,
) -> anyhow::Result<()> {
    let request_id = offload_req.request_id.clone();
    let operation_id = offload_req.operation_id;
    let offload_span = offload_req.traceparent.as_deref()
        .map(|tp| dynamo_runtime::logging::make_linked_span("kvbm.offload", tp))
        .unwrap_or_else(|| tracing::info_span!("kvbm.offload",
            request_id = %request_id,
            operation_id = %operation_id,
            num_blocks = offload_req.block_ids.len(),
            otel.name = "kvbm.offload",
            description = "Device-to-host offload (GPU VRAM to pinned RAM)",
        ));
    _process_offload_request_inner(offload_req, block_manager, leader, kvbm_metrics, drain_tx)
        .instrument(offload_span)
        .await
}

async fn _process_offload_request_inner(
    offload_req: LocalOffloadRequest,
    block_manager: &VllmBlockManager,
    leader: &Arc<KvbmLeader>,
    kvbm_metrics: KvbmMetrics,
    drain_tx: &mpsc::UnboundedSender<DrainItem>,
) -> anyhow::Result<()> {
    let request_id = offload_req.request_id.clone();
    let operation_id = offload_req.operation_id;

    let bypass_cpu_mem = should_bypass_cpu_cache();
    if bypass_cpu_mem {
        kvbm_metrics
            .offload_blocks_d2d
            .inc_by(offload_req.block_ids.len() as u64);
        process_offload_to_storage(
            offload_req,
            block_manager.disk().unwrap(),
            BlockTransferPool::Disk,
            leader,
            &request_id,
            &operation_id,
            "disk",
            None,
        )
        .await?;
    } else {
        kvbm_metrics
            .offload_blocks_d2h
            .inc_by(offload_req.block_ids.len() as u64);
        process_offload_to_storage(
            offload_req,
            block_manager.host().unwrap(),
            BlockTransferPool::Host,
            leader,
            &request_id,
            &operation_id,
            "host",
            Some(drain_tx),
        )
        .await?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
async fn process_offload_to_storage<S, L, M>(
    offload_req: LocalOffloadRequest,
    storage_pool: &dyn BlockPool<S, L, M>,
    transfer_pool: BlockTransferPool,
    leader: &Arc<KvbmLeader>,
    _request_id: &str,
    _operation_id: &uuid::Uuid,
    _storage_name: &str,
    drain_tx: Option<&mpsc::UnboundedSender<DrainItem>>,
) -> anyhow::Result<()>
where
    S: Storage + NixlRegisterableStorage + 'static,
    L: LocalityProvider + 'static,
    M: BlockMetadata + 'static,
{
    let blocks = {
        let _alloc_span = tracing::info_span!(
            "bounce_alloc",
            otel.name = "kvbm.bounce_alloc",
            description = "Allocate pinned host bounce buffers for offload",
            num_blocks = offload_req.block_ids.len(),
        ).entered();
        tokio::task::block_in_place(|| {
            storage_pool.allocate_blocks_blocking(offload_req.block_ids.len())
        })?
    };
    let token_blocks = offload_req.token_blocks;
    let allocated_block_ids: Vec<usize> = blocks.iter().map(|b| b.block_id()).collect();
    let block_pairs: Vec<(usize, usize)> = offload_req
        .block_ids
        .into_iter()
        .zip(allocated_block_ids.into_iter())
        .collect();

    let mut blocks_to_register = Vec::new();
    let priorities = offload_req.priorities;
    for ((mut mutable_block, token_block), priority) in blocks
        .into_iter()
        .zip(token_blocks.into_iter())
        .zip(priorities.into_iter())
    {
        mutable_block
            .apply_token_block(token_block.clone())
            .map_err(|e| anyhow::anyhow!("failed to apply token block: {:?}", e))?;
        let updated_metadata = mutable_block.metadata().with_priority(priority);
        mutable_block.update_metadata(updated_metadata);
        blocks_to_register.push(mutable_block);
    }

    let sequence_hashes =
        if transfer_pool == BlockTransferPool::Host && leader.remote_registry_enabled() {
            Some(offload_req.sequence_hashes.clone())
        } else {
            None
        };
    let block_xfer_req = BlockTransferRequest {
        from_pool: BlockTransferPool::Device,
        to_pool: transfer_pool,
        blocks: block_pairs,
        connector_req: Some(LeaderTransferRequest {
            key: offload_req.key.clone(),
            uuid: offload_req.operation_id,
            requirement: None,
            request_type: RequestType::Scheduled,
            chained: false,
        }),
        sequence_hashes,
        traceparent: None,
    };
    let notify_receiver = leader.transfer_blocks_request(block_xfer_req).await?;
    notify_receiver
        .await
        .map_err(|_| anyhow::anyhow!("offload transfer completion failed"))?;

    let immutable_blocks = storage_pool.register_blocks(blocks_to_register).await?;
    let is_host_transfer = transfer_pool == BlockTransferPool::Host;
    if is_host_transfer && leader.remote_registry_enabled() {
        if let Some(drain_tx) = drain_tx {
            let host_block_ids = immutable_blocks.iter().map(|b| b.block_id()).collect();
            let pin_guard = crate::block_manager::pool::PinGuard::new(immutable_blocks);
            let item = DrainItem {
                key: offload_req.key.clone(),
                request_id: offload_req.request_id.clone(),
                sequence_hashes: offload_req.sequence_hashes.clone(),
                host_block_ids,
                pin_guard,
                block_size: offload_req.block_size,
                traceparent: offload_req.traceparent.clone(),
                baggage: offload_req.baggage.clone(),
            };
            let _ = drain_tx.send(item);
            return Ok(());
        }
    }
    drop(immutable_blocks);
    Ok(())
}

async fn process_onboard_request(
    onboard_req: LocalOnboardRequest,
    leader: &Arc<KvbmLeader>,
    kvbm_metrics: KvbmMetrics,
) -> anyhow::Result<()> {
    let onboard_span = onboard_req.traceparent.as_deref()
        .map(|tp| dynamo_runtime::logging::make_linked_span("kvbm.onboard.from_host", tp))
        .unwrap_or_else(|| tracing::info_span!("kvbm.onboard.from_host",
            request_id = %onboard_req.request_id,
            operation_id = %onboard_req.operation_id,
            num_blocks = onboard_req.src_blocks.len(),
            src_pool = ?onboard_req.src_blocks.storage_pool(),
            otel.name = "kvbm.onboard.from_host",
            description = "Onboard from host cache hit (H2D DMA only, no disk read)",
        ));
    _process_onboard_request_inner(onboard_req, leader, kvbm_metrics)
        .instrument(onboard_span)
        .await
}

async fn _process_onboard_request_inner(
    onboard_req: LocalOnboardRequest,
    leader: &Arc<KvbmLeader>,
    kvbm_metrics: KvbmMetrics,
) -> anyhow::Result<()> {
    if onboard_req.src_blocks.storage_pool() == BlockTransferPool::Host {
        kvbm_metrics
            .onboard_blocks_h2d
            .inc_by(onboard_req.src_blocks.len() as u64);
    } else if onboard_req.src_blocks.storage_pool() == BlockTransferPool::Disk {
        kvbm_metrics
            .onboard_blocks_d2d
            .inc_by(onboard_req.src_blocks.len() as u64);
    }

    let src_block_ids = onboard_req.src_blocks.block_ids();
    let block_pairs = src_block_ids
        .iter()
        .zip(onboard_req.dst_block_ids.iter())
        .map(|(src, dst)| (*src, *dst))
        .collect::<Vec<_>>();
    let block_xfer_req = BlockTransferRequest {
        from_pool: onboard_req.src_blocks.storage_pool(),
        to_pool: BlockTransferPool::Device,
        blocks: block_pairs,
        connector_req: Some(LeaderTransferRequest {
            key: onboard_req.key.clone(),
            uuid: onboard_req.operation_id,
            requirement: None,
            request_type: RequestType::Immediate,
            chained: false,
        }),
        sequence_hashes: None,
        traceparent: onboard_req.traceparent.clone(),
    };
    let notify_receiver = leader
        .transfer_blocks_request(block_xfer_req)
        .await?;
    notify_receiver
        .await
        .map_err(|_| anyhow::anyhow!("onboarding transfer completion failed"))?;
    Ok(())
}

async fn process_remote_transfer_request(
    req: RemoteTransferRequest,
    block_manager: &VllmBlockManager,
    leader: &Arc<KvbmLeader>,
    kvbm_metrics: KvbmMetrics,
    pin_registry: &PinRegistry,
) -> anyhow::Result<()> {
    let request_id = &req.request_id;
    let operation_id = &req.operation_id;
    let pin_id = req.pin_id;
    let onboard_token_blocks = req.token_blocks.clone();
    let process_span = req
        .traceparent
        .as_deref()
        .map(|tp| dynamo_runtime::logging::make_linked_span("kvbm.remote_transfer", tp))
        .unwrap_or_else(|| {
            tracing::info_span!(
                "kvbm.remote_transfer",
                request_id = %request_id,
                operation_id = %operation_id,
                is_onboard = req.is_onboard,
                num_blocks = req.sequence_hashes.len(),
                otel.name = "kvbm.remote_transfer",
                description = "Leader-side remote transfer orchestration (NIXL + H2D pipeline)",
            )
        });

    let release_pin = |pin_registry: &PinRegistry, pin_id: Option<uuid::Uuid>| {
        if let Some(guard) = pin_id.and_then(|id| pin_registry.remove(&id)) {
            tracing::debug!(pin_id = ?pin_id, num_blocks = guard.count(), "released pin guard");
        }
    };

    // Intentionally do not deduplicate concurrent G4 onboard requests.
    // The previous inflight wait/registration path serialized overlapping
    // requests at request granularity, which prevented true concurrent
    // cold-path onboarding. We accept duplicate remote reads here to keep
    // the transfer path concurrent.

    let (hashes_with_positions, filtered_host_ids) = if let Some(handle) = leader.remote_handle() {
        match vllm_int::filter_for_offload(
            &handle,
            &req.sequence_hashes,
            req.host_block_ids.as_deref(),
            leader.worker_id(),
            req.is_onboard,
        )
        .await
        {
            Some(filtered) => filtered,
            None => {
                release_pin(pin_registry, pin_id);
                return Ok(());
            }
        }
    } else {
        let hashes_with_positions: Vec<(u64, u32)> = req
            .sequence_hashes
            .iter()
            .enumerate()
            .map(|(pos, &hash)| (hash, pos as u32))
            .collect();
        (hashes_with_positions, req.host_block_ids.clone())
    };

    let num_blocks = hashes_with_positions.len();
    let storage_config = match leader.remote_storage_config() {
        Some(cfg) => cfg,
        None => {
            tracing::warn!(
                request_id = %request_id,
                "No remote storage configured (check DYN_KVBM_REMOTE_STORAGE_TYPE, \
                 DYN_KVBM_REMOTE_DISK_PATH(S), or DYN_KVBM_OBJECT_BUCKET). \
                 Cannot execute remote transfer."
            );
            release_pin(pin_registry, pin_id);
            return Err(anyhow::anyhow!(
                "Remote storage not configured — set DYN_KVBM_REMOTE_DISK_PATH(S) or DYN_KVBM_OBJECT_BUCKET"
            ));
        }
    };
    let backend_label = match &storage_config {
        RemoteStorageConfig::Object { .. } => "object",
        RemoteStorageConfig::Disk { transfer_flags, .. } => {
            use crate::block_manager::config::DISK_FLAG_GDS_WRITE;
            if transfer_flags & DISK_FLAG_GDS_WRITE != 0 {
                "gds_mt"
            } else {
                "posix"
            }
        }
    };

    let hashes: Vec<u64> = hashes_with_positions.iter().map(|&(h, _)| h).collect();
    let _alloc_span = tracing::info_span!(
        parent: process_span.clone(),
        "kvbm.remote_transfer_allocate",
        request_id = %request_id,
        operation_id = %operation_id,
        is_onboard = req.is_onboard,
        num_blocks = num_blocks,
        otel.name = "kvbm.g4_allocate",
        description = "Allocate host bounce buffers for remote transfer",
    )
    .entered();
    let (bounce, device, onboard_host_blocks) = if req.is_h2o() {
        let bounce = filtered_host_ids
            .ok_or_else(|| anyhow::anyhow!("H2R transfer requires host_block_ids"))?;
        (bounce, vec![], None)
    } else {
        let host_pool = block_manager
            .host()
            .ok_or_else(|| anyhow::anyhow!("Host pool not available for bounce buffers"))?;
        let host_blocks =
            tokio::task::block_in_place(|| host_pool.allocate_blocks_blocking(num_blocks))?;
        let bounce = host_blocks.iter().map(|b| b.block_id()).collect();
        let device = req.device_block_ids.iter().copied().collect();
        (bounce, device, Some(host_blocks))
    };
    drop(_alloc_span);

    let _pipeline_span = tracing::info_span!(
        parent: process_span.clone(),
        "kvbm.remote_transfer_build_pipeline",
        request_id = %request_id,
        operation_id = %operation_id,
        is_onboard = req.is_onboard,
        num_blocks = num_blocks,
        otel.name = "kvbm.g4_build_pipeline",
        description = "Build NIXL transfer pipeline (descriptors + storage config)",
    )
    .entered();
    let pipeline = vllm_int::create_transfer_pipeline(
        &hashes,
        &storage_config,
        req.block_size,
        leader.worker_id() as usize,
        leader.world_size(),
        req.is_onboard,
        req.is_h2o(),
        bounce,
        device,
    );
    drop(_pipeline_span);

    let is_chained = !req.is_onboard;
    let mut wire_req =
        crate::block_manager::distributed::RemoteTransferRequest::new_with_connector_req(
            req.request_id.clone(),
            req.operation_id,
            &pipeline,
            LeaderTransferRequest {
                key: req.key.clone(),
                uuid: *operation_id,
                requirement: None,
                request_type: RequestType::Immediate,
                chained: is_chained,
            },
        );
    wire_req.traceparent = req.traceparent.clone();
    let dispatch_span = tracing::info_span!(
        parent: &process_span,
        "kvbm.remote_transfer_dispatch",
        request_id = %request_id,
        operation_id = %operation_id,
        is_onboard = req.is_onboard,
        num_blocks = num_blocks,
        otel.name = "kvbm.g4_dispatch",
        description = "ZMQ dispatch of remote transfer request to workers",
    );
    let notify_receiver = leader
        .remote_transfer_request(wire_req)
        .instrument(dispatch_span)
        .await?;
    let transfer_start = Instant::now();
    let transfer_bytes = (num_blocks as u64).saturating_mul(req.block_size as u64);

    let g4_timeout = g4_transfer_timeout();
    let result = match tokio::time::timeout(g4_timeout, notify_receiver).await {
        Ok(Ok(_)) => {
            crate::record_remote_metrics!(
                kvbm_metrics,
                req.is_onboard,
                num_blocks,
                transfer_bytes,
                "success",
                backend_label,
                transfer_start.elapsed().as_secs_f64()
            );

            if !req.is_onboard {
                if let Some(handle) = leader.remote_handle() {
                    vllm_int::register_tp(
                        &handle,
                        &hashes_with_positions,
                        &storage_config,
                        leader.world_size(),
                    )
                    .await;
                }
            }

            if req.is_onboard
                && let (Some(host_blocks), Some(token_blocks)) =
                    (onboard_host_blocks, onboard_token_blocks)
                && let Some(host_pool) = block_manager.host()
            {
                let mut blocks_to_register = Vec::new();
                for (mut block, token_block) in host_blocks.into_iter().zip(token_blocks) {
                    if block.apply_token_block(token_block).is_ok() {
                        blocks_to_register.push(block);
                    }
                }
                if !blocks_to_register.is_empty() {
                    let _ = host_pool.register_blocks(blocks_to_register).await;
                }
            }
            Ok(())
        }
        Ok(Err(_)) => {
            crate::record_remote_metrics!(
                kvbm_metrics,
                req.is_onboard,
                num_blocks,
                transfer_bytes,
                "error",
                backend_label,
                transfer_start.elapsed().as_secs_f64()
            );
            Err(anyhow::anyhow!(
                "Remote transfer ({}) completion notification failed: request_id={}, num_blocks={}, backend={}",
                if req.is_onboard { "onboard/read" } else { "offload/write" },
                req.request_id,
                num_blocks,
                backend_label,
            ))
        }
        Err(_) => {
            crate::record_remote_metrics!(
                kvbm_metrics,
                req.is_onboard,
                num_blocks,
                transfer_bytes,
                "timeout",
                backend_label,
                transfer_start.elapsed().as_secs_f64()
            );
            Err(anyhow::anyhow!(
                "Remote transfer ({}) timed out after {} seconds: request_id={}, num_blocks={}, backend={}",
                if req.is_onboard { "onboard/read" } else { "offload/write" },
                g4_timeout.as_secs(),
                req.request_id,
                num_blocks,
                backend_label,
            ))
        }
    };

    release_pin(pin_registry, pin_id);
    result
}
