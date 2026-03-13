// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{sync::Arc, time::Instant};

use dynamo_runtime::config::environment_names::kvbm::remote_storage as env_g4;
use dynamo_runtime::utils::task::CriticalTaskExecutionHandle;
use once_cell::sync::Lazy;
use tokio::{runtime::Handle, sync::mpsc};
use tokio::task::JoinSet;
use tokio_util::sync::CancellationToken;
use tracing::Instrument;

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
    pool::{BlockPoolError, PinRegistry},
    transfer_orchestrator::{TransferPriority, priority_channel, run_priority_worker},
};
use crate::block_manager::KvBlockManager;

use super::{DrainItem, LocalOffloadRequest, LocalOnboardRequest, LocalTransferRequest, RemoteTransferRequest};

type VllmBlockManager =
    KvBlockManager<Logical<DistributedLeaderWorkerResources>, BasicMetadata>;

const DEFAULT_DRAIN_QUEUE_CAP: usize = 512;
const DEFAULT_REMOTE_HIGH_QUEUE_CAP: usize = 256;
const DEFAULT_REMOTE_LOW_QUEUE_CAP: usize = 512;
const DEFAULT_G4_TRANSFER_TIMEOUT_SECS: u64 = 30;

static G4_TRANSFER_TIMEOUT: Lazy<std::time::Duration> = Lazy::new(|| {
    let secs: u64 = std::env::var(env_g4::DYN_KVBM_G4_TRANSFER_TIMEOUT_SECS)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_G4_TRANSFER_TIMEOUT_SECS);
    std::time::Duration::from_secs(secs)
});

static DRAIN_QUEUE_CAP: Lazy<usize> = Lazy::new(|| {
    std::env::var("DYN_KVBM_G4_DRAIN_QUEUE_CAP")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_DRAIN_QUEUE_CAP)
});

static REMOTE_HIGH_QUEUE_CAP: Lazy<usize> = Lazy::new(|| {
    std::env::var("DYN_KVBM_G4_REMOTE_HIGH_QUEUE_CAP")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_REMOTE_HIGH_QUEUE_CAP)
});

static REMOTE_LOW_QUEUE_CAP: Lazy<usize> = Lazy::new(|| {
    std::env::var("DYN_KVBM_G4_REMOTE_LOW_QUEUE_CAP")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_REMOTE_LOW_QUEUE_CAP)
});

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
    ) -> anyhow::Result<()> {
        let (onboard_tx, mut onboard_rx) = mpsc::unbounded_channel::<LocalOnboardRequest>();
        let (offload_tx, mut offload_rx) = mpsc::unbounded_channel::<LocalOffloadRequest>();
        let (remote_onboard_tx, remote_onboard_rx) = priority_channel::<RemoteTransferRequest>(
            *REMOTE_HIGH_QUEUE_CAP,
            *REMOTE_LOW_QUEUE_CAP,
        );
        let (remote_offload_tx, remote_offload_rx) = priority_channel::<RemoteTransferRequest>(
            *REMOTE_HIGH_QUEUE_CAP,
            *REMOTE_LOW_QUEUE_CAP,
        );
        let (drain_tx, mut drain_rx) = mpsc::unbounded_channel::<DrainItem>();

        let pin_registry = PinRegistry::new();
        let pin_registry_drain = pin_registry.clone();
        let pin_registry_remote_onboard = pin_registry.clone();
        let pin_registry_remote_offload = pin_registry.clone();

        let block_manager_offload = self.block_manager.clone();
        let block_manager_remote_onboard = self.block_manager.clone();
        let block_manager_remote_offload = self.block_manager.clone();
        let leader_offload = Arc::clone(&self.leader);
        let leader_onboard = Arc::clone(&self.leader);
        let leader_remote_onboard = Arc::clone(&self.leader);
        let leader_remote_offload = Arc::clone(&self.leader);
        let drain_tx_for_offload = drain_tx.clone();
        let remote_offload_tx_for_drain = remote_offload_tx.clone();

        let kvbm_metrics_onboard = kvbm_metrics.clone();
        let kvbm_metrics_offload = kvbm_metrics.clone();
        let kvbm_metrics_remote_onboard = kvbm_metrics.clone();
        let kvbm_metrics_remote_offload = kvbm_metrics.clone();

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
                                    join_set.spawn(async move {
                                        if let Err(e) = process_onboard_request(req, &leader, metrics).await {
                                            tracing::error!("LocalOnboardTask error: {:?}", e);
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
                                    let request_id = req.request_id.clone();
                                    let operation_id = req.operation_id;
                                    let block_manager = block_manager_offload.clone();
                                    let leader = Arc::clone(&leader_offload);
                                    let metrics = kvbm_metrics_offload.clone();
                                    let drain_tx = drain_tx_for_offload.clone();

                                    join_set.spawn(async move {
                                        if let Err(e) = process_offload_request(
                                            req,
                                            &block_manager,
                                            &leader,
                                            metrics,
                                            &drain_tx,
                                        )
                                        .await
                                        {
                                            tracing::error!("LocalOffloadTask error: {:?}", e);
                                            let fake_xfer = BlockTransferRequest {
                                                from_pool: BlockTransferPool::Device,
                                                to_pool: BlockTransferPool::Host,
                                                blocks: vec![],
                                                connector_req: Some(LeaderTransferRequest {
                                                    request_id: request_id.clone(),
                                                    uuid: operation_id,
                                                    requirement: None,
                                                    request_type: RequestType::Immediate,
                                                    chained: false,
                                                }),
                                                sequence_hashes: None,
                                            };
                                            if let Ok(notify_receiver) = leader.transfer_blocks_request(fake_xfer).await {
                                                let _ = notify_receiver.await;
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

        let remote_onboard_task = CriticalTaskExecutionHandle::new_with_runtime(
            |cancellation_token_remote| async move {
                run_priority_worker(
                    cancellation_token_remote,
                    remote_onboard_rx,
                    move |req| {
                        let block_manager = block_manager_remote_onboard.clone();
                        let leader = Arc::clone(&leader_remote_onboard);
                        let metrics = kvbm_metrics_remote_onboard.clone();
                        let pin_reg = pin_registry_remote_onboard.clone();
                        async move {
                            let request_id = req.request_id.clone();
                            if let Err(e) = process_remote_transfer_request(
                                req,
                                &block_manager,
                                &leader,
                                metrics,
                                &pin_reg,
                            )
                            .await
                            {
                                tracing::error!("RemoteTransferTask error: {:?}", e);
                                if matches!(
                                    e.downcast_ref::<BlockPoolError>(),
                                    Some(BlockPoolError::NotEnoughBlocksAvailable(_, _))
                                ) {
                                    leader.mark_g4_failed_skip_retry(&request_id);
                                } else {
                                    leader.mark_g4_failed(&request_id);
                                }
                            }
                        }
                    },
                )
                .await;
                Ok(())
            },
            task_token.clone(),
            "RemoteOnboardTransferTask",
            &task_handle,
        )
        .unwrap();

        let remote_offload_task = CriticalTaskExecutionHandle::new_with_runtime(
            |cancellation_token_remote| async move {
                run_priority_worker(
                    cancellation_token_remote,
                    remote_offload_rx,
                    move |req| {
                        let block_manager = block_manager_remote_offload.clone();
                        let leader = Arc::clone(&leader_remote_offload);
                        let metrics = kvbm_metrics_remote_offload.clone();
                        let pin_reg = pin_registry_remote_offload.clone();
                        async move {
                            let request_id = req.request_id.clone();
                            if let Err(e) = process_remote_transfer_request(
                                req,
                                &block_manager,
                                &leader,
                                metrics,
                                &pin_reg,
                            )
                            .await
                            {
                                tracing::error!("RemoteTransferTask error: {:?}", e);
                                if matches!(
                                    e.downcast_ref::<BlockPoolError>(),
                                    Some(BlockPoolError::NotEnoughBlocksAvailable(_, _))
                                ) {
                                    leader.mark_g4_failed_skip_retry(&request_id);
                                } else {
                                    leader.mark_g4_failed(&request_id);
                                }
                            }
                        }
                    },
                )
                .await;
                Ok(())
            },
            task_token.clone(),
            "RemoteOffloadTransferTask",
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

                    let drain_wait_ms = item.enqueued_at.elapsed().as_millis();
                    let h2o_operation_id = uuid::Uuid::new_v4();
                    pin_registry_drain.insert(h2o_operation_id, item.pin_guard);
                    let request_id = item.request_id.clone();
                    let num_blocks = item.sequence_hashes.len();

                    let h2o_req = RemoteTransferRequest::new_h2o(
                        item.request_id,
                        item.sequence_hashes,
                        item.host_block_ids,
                        h2o_operation_id,
                        item.block_size,
                        h2o_operation_id,
                        item.traceparent,
                    );

                    if remote_offload_tx_for_drain
                        .send(TransferPriority::Low, h2o_req)
                        .await
                        .is_err()
                    {
                        pin_registry_drain.remove(&h2o_operation_id);
                        continue;
                    }
                    tracing::info!(
                        target: "kvbm-g4",
                        request_id = %request_id,
                        operation_id = %h2o_operation_id,
                        num_blocks,
                        drain_wait_ms,
                        "background H2O drain dispatched"
                    );
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
                                let _ = remote_onboard_tx.send(TransferPriority::High, remote_req).await;
                            } else {
                                let _ = remote_offload_tx.send(TransferPriority::Low, remote_req).await;
                            }
                        }
                        None => break,
                    }
                }
            }
        }

        drop(onboard_tx);
        drop(offload_tx);
        drop(remote_onboard_tx);
        drop(remote_offload_tx);
        onboard_task.cancel();
        offload_task.cancel();
        remote_onboard_task.cancel();
        remote_offload_task.cancel();
        drain_task.cancel();
        let _ = onboard_task.join().await;
        let _ = offload_task.join().await;
        let _ = remote_onboard_task.join().await;
        let _ = remote_offload_task.join().await;
        let _ = drain_task.join().await;
        Ok(())
    }
}

#[tracing::instrument(level = "info", skip_all, fields(
    request_id = %offload_req.request_id,
    operation_id = %offload_req.operation_id,
    num_blocks = offload_req.block_ids.len(),
    otel.name = "kvbm.offload",
))]
async fn process_offload_request(
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
    let allocate_start = Instant::now();
    let blocks = tokio::task::block_in_place(|| {
        storage_pool.allocate_blocks_blocking(offload_req.block_ids.len())
    })?;
    tracing::info!(
        target: "kvbm-g4",
        request_id = %offload_req.request_id,
        operation_id = %offload_req.operation_id,
        transfer_pool = ?transfer_pool,
        num_blocks = offload_req.block_ids.len(),
        elapsed_ms = allocate_start.elapsed().as_millis(),
        "offload storage allocation complete"
    );
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

    let sequence_hashes = if transfer_pool == BlockTransferPool::Host && leader.remote_registry_enabled() {
        Some(offload_req.sequence_hashes.clone())
    } else {
        None
    };
    let block_xfer_req = BlockTransferRequest {
        from_pool: BlockTransferPool::Device,
        to_pool: transfer_pool,
        blocks: block_pairs,
        connector_req: Some(LeaderTransferRequest {
            request_id: offload_req.request_id.clone(),
            uuid: offload_req.operation_id,
            requirement: None,
            request_type: RequestType::Scheduled,
            chained: false,
        }),
        sequence_hashes,
    };
    let notify_receiver = leader.transfer_blocks_request(block_xfer_req).await?;
    let transfer_start = Instant::now();
    notify_receiver
        .await
        .map_err(|_| anyhow::anyhow!("offload transfer completion failed"))?;
    tracing::info!(
        target: "kvbm-g4",
        request_id = %offload_req.request_id,
        operation_id = %offload_req.operation_id,
        transfer_pool = ?transfer_pool,
        num_blocks = offload_req.sequence_hashes.len(),
        elapsed_ms = transfer_start.elapsed().as_millis(),
        "offload transfer completion notification received"
    );

    let register_start = Instant::now();
    let immutable_blocks = storage_pool.register_blocks(blocks_to_register).await?;
    tracing::info!(
        target: "kvbm-g4",
        request_id = %offload_req.request_id,
        operation_id = %offload_req.operation_id,
        transfer_pool = ?transfer_pool,
        num_blocks = immutable_blocks.len(),
        elapsed_ms = register_start.elapsed().as_millis(),
        "offload storage registration complete"
    );
    let is_host_transfer = transfer_pool == BlockTransferPool::Host;
    if is_host_transfer && leader.remote_registry_enabled() {
        if let Some(drain_tx) = drain_tx {
            let host_block_ids = immutable_blocks.iter().map(|b| b.block_id()).collect();
            let pin_guard = crate::block_manager::pool::PinGuard::new(immutable_blocks);
            let item = DrainItem {
                request_id: offload_req.request_id.clone(),
                sequence_hashes: offload_req.sequence_hashes.clone(),
                host_block_ids,
                pin_guard,
                block_size: offload_req.block_size,
                traceparent: offload_req.traceparent.clone(),
                enqueued_at: Instant::now(),
            };
            tracing::info!(
                target: "kvbm-g4",
                request_id = %item.request_id,
                operation_id = %offload_req.operation_id,
                num_blocks = item.sequence_hashes.len(),
                "enqueuing host blocks for background H2O drain"
            );
            let _ = drain_tx.send(item);
            return Ok(());
        }
    }
    drop(immutable_blocks);
    Ok(())
}

#[tracing::instrument(level = "info", skip_all, fields(
    request_id = %onboard_req.request_id,
    operation_id = %onboard_req.operation_id,
    num_blocks = onboard_req.src_blocks.len(),
    src_pool = ?onboard_req.src_blocks.storage_pool(),
    otel.name = "kvbm.onboard_local",
))]
async fn process_onboard_request(
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
            request_id: onboard_req.request_id.clone(),
            uuid: onboard_req.operation_id,
            requirement: None,
            request_type: RequestType::Immediate,
            chained: false,
        }),
        sequence_hashes: None,
    };
    let notify_receiver = leader.transfer_blocks_request(block_xfer_req).await?;
    notify_receiver
        .await
        .map_err(|_| anyhow::anyhow!("onboarding transfer completion failed"))?;
    Ok(())
}

#[tracing::instrument(level = "info", skip_all, fields(
    request_id = %req.request_id,
    operation_id = %req.operation_id,
    is_onboard = req.is_onboard,
    num_blocks = req.sequence_hashes.len(),
    otel.name = "kvbm.process_remote_transfer",
))]
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
        .map(|tp| dynamo_runtime::logging::make_linked_span("kvbm.process_remote_transfer", tp))
        .unwrap_or_else(|| {
            tracing::info_span!(
                "kvbm.process_remote_transfer",
                request_id = %request_id,
                operation_id = %operation_id,
                is_onboard = req.is_onboard,
                num_blocks = req.sequence_hashes.len(),
                otel.name = "kvbm.process_remote_transfer",
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
        let filter_start = Instant::now();
        match vllm_int::filter_for_offload(
            &handle,
            &req.sequence_hashes,
            req.host_block_ids.as_deref(),
            leader.worker_id(),
            req.is_onboard,
        )
        .await
        {
            Some(filtered) => {
                tracing::info!(
                    target: "kvbm-g4",
                    request_id = %request_id,
                    operation_id = %operation_id,
                    is_onboard = req.is_onboard,
                    requested_blocks = req.sequence_hashes.len(),
                    filtered_blocks = filtered.0.len(),
                    elapsed_ms = filter_start.elapsed().as_millis(),
                    "remote registry offload filter complete"
                );
                filtered
            }
            None => {
                tracing::info!(
                    target: "kvbm-g4",
                    request_id = %request_id,
                    operation_id = %operation_id,
                    is_onboard = req.is_onboard,
                    requested_blocks = req.sequence_hashes.len(),
                    elapsed_ms = filter_start.elapsed().as_millis(),
                    "remote registry offload filter skipped all blocks"
                );
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

    let (bounce, device, onboard_host_blocks) = {
        let _alloc_span = tracing::info_span!(
            parent: process_span.clone(),
            "kvbm.remote_transfer_allocate",
            request_id = %request_id,
            operation_id = %operation_id,
            is_onboard = req.is_onboard,
            num_blocks = num_blocks,
            otel.name = "kvbm.remote_transfer_allocate",
        )
        .entered();

        if req.is_h2o() {
            let bounce = filtered_host_ids
                .ok_or_else(|| anyhow::anyhow!("H2R transfer requires host_block_ids"))?;
            (bounce, vec![], None)
        } else {
            let host_pool = block_manager
                .host()
                .ok_or_else(|| anyhow::anyhow!("Host pool not available for bounce buffers"))?;
            let alloc_start = Instant::now();
            let available_before = host_pool.available_blocks();
            if available_before < num_blocks as u64 {
                tracing::warn!(
                    target: "kvbm-g4",
                    request_id = %request_id,
                    operation_id = %operation_id,
                    is_onboard = req.is_onboard,
                    num_blocks,
                    block_size = req.block_size,
                    available_before,
                    "insufficient host bounce capacity for G4 transfer; skipping to recompute"
                );
                return Err(BlockPoolError::NotEnoughBlocksAvailable(
                    num_blocks,
                    available_before as usize,
                )
                .into());
            }
            let host_blocks =
                tokio::task::block_in_place(|| host_pool.allocate_blocks_blocking(num_blocks))?;
            tracing::info!(
                target: "kvbm-g4",
                request_id = %request_id,
                operation_id = %operation_id,
                is_onboard = req.is_onboard,
                num_blocks,
                block_size = req.block_size,
                available_before,
                available_after = host_pool.available_blocks(),
                elapsed_ms = alloc_start.elapsed().as_millis(),
                "host bounce block allocation complete"
            );
            let bounce = host_blocks.iter().map(|b| b.block_id()).collect();
            let device = req.device_block_ids.iter().copied().collect();
            (bounce, device, Some(host_blocks))
        }
    };

    let _pipeline_span = tracing::info_span!(
        parent: process_span.clone(),
        "kvbm.remote_transfer_build_pipeline",
        request_id = %request_id,
        operation_id = %operation_id,
        is_onboard = req.is_onboard,
        num_blocks = num_blocks,
        otel.name = "kvbm.remote_transfer_build_pipeline",
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

    let is_host_prefetch_only = req.is_onboard && req.device_block_ids.is_empty();
    let is_chained = !req.is_onboard;
    let mut wire_req = if is_host_prefetch_only {
        crate::block_manager::distributed::RemoteTransferRequest::new(
            req.request_id.clone(),
            req.operation_id,
            &pipeline,
        )
    } else {
        crate::block_manager::distributed::RemoteTransferRequest::new_with_connector_req(
            req.request_id.clone(),
            req.operation_id,
            &pipeline,
            LeaderTransferRequest {
                request_id: request_id.clone(),
                uuid: *operation_id,
                requirement: None,
                request_type: RequestType::Immediate,
                chained: is_chained,
            },
        )
    };
    wire_req.traceparent = req.traceparent.clone();
    let dispatch_span = tracing::info_span!(
        parent: &process_span,
        "kvbm.remote_transfer_dispatch",
        request_id = %request_id,
        operation_id = %operation_id,
        is_onboard = req.is_onboard,
        num_blocks = num_blocks,
        otel.name = "kvbm.remote_transfer_dispatch",
    );
    let notify_receiver = leader
        .remote_transfer_request(wire_req)
        .instrument(dispatch_span)
        .await?;
    let transfer_start = Instant::now();
    let transfer_bytes = (num_blocks as u64).saturating_mul(req.block_size as u64);

    let result = match tokio::time::timeout(*G4_TRANSFER_TIMEOUT, notify_receiver).await {
        Ok(Ok(_)) => {
            tracing::info!(
                target: "kvbm-g4",
                request_id = %request_id,
                operation_id = %operation_id,
                is_onboard = req.is_onboard,
                num_blocks,
                transfer_bytes,
                backend = backend_label,
                elapsed_ms = transfer_start.elapsed().as_millis(),
                "remote transfer completion notification received"
            );
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
                    let publish_start = Instant::now();
                    vllm_int::register_tp(
                        &handle,
                        &hashes_with_positions,
                        &storage_config,
                        leader.world_size(),
                    )
                    .await;
                    tracing::info!(
                        target: "kvbm-g4",
                        request_id = %request_id,
                        operation_id = %operation_id,
                        num_blocks = hashes_with_positions.len(),
                        elapsed_ms = publish_start.elapsed().as_millis(),
                        "remote registry publication complete"
                    );
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
                    let register_count = blocks_to_register.len();
                    let register_start = Instant::now();
                    let _ = host_pool.register_blocks(blocks_to_register).await;
                    tracing::info!(
                        target: "kvbm-g4",
                        request_id = %request_id,
                        operation_id = %operation_id,
                        num_blocks = register_count,
                        block_size = req.block_size,
                        elapsed_ms = register_start.elapsed().as_millis(),
                        available_after = host_pool.available_blocks(),
                        "host onboarding block registration complete"
                    );
                }
                if req.device_block_ids.is_empty() {
                    leader.set_g4_prefetch_prefix(request_id, hashes.len());
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
            Err(anyhow::anyhow!("Remote transfer completion notification failed"))
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
                "Remote transfer timed out after {} seconds",
                G4_TRANSFER_TIMEOUT.as_secs()
            ))
        }
    };

    release_pin(pin_registry, pin_id);
    result
}
