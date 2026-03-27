// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Effect executor: bridge between the pure `apply()` reducer and side-effectful I/O.
//!
//! The slot state machine produces `Vec<SlotEffect<H, D>>` from each `apply()` call.
//! This module consumes those effects and performs the actual work (transfers, cache
//! stats, diagnostics).  The executor never modifies `slot.phase` -- that is the
//! reducer's exclusive responsibility.

use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

use tokio::sync::mpsc;

use crate::block_manager::{
    BasicMetadata, DiskStorage, ImmutableBlock, KvBlockManager, PinnedStorage,
    block::{
        data::logical::distributed_leader_worker::DistributedLeaderWorkerResources,
        locality::Logical,
    },
    connector::{
        cache_stats::CacheStatsTracker,
        protocol::{RequestType, SlotKey, TransferType, WorkerTransferRequest},
    },
};
use crate::block_manager::distributed::KvbmLeader;

use super::Slot; // bring trait methods (request_id, sequence) into scope
use super::slot_machine::{DiagLevel, SlotContext, SlotEffect, SlotEvent};
use super::{
    AnyImmutableBlocks, LocalOnboardRequest, LocalOffloadRequest,
    LocalTransferRequest, RemoteTransferRequest,
};
use crate::tokens::TokenBlock;

type VllmBlockManager = KvBlockManager<Logical<DistributedLeaderWorkerResources>, BasicMetadata>;
type VllmLocality = Logical<DistributedLeaderWorkerResources>;
type VllmHostBlocks = Vec<ImmutableBlock<PinnedStorage, VllmLocality, BasicMetadata>>;
type VllmDiskBlocks = Vec<ImmutableBlock<DiskStorage, VllmLocality, BasicMetadata>>;

/// Infrastructure context needed to execute side effects.
///
/// Passed to the executor by the adapter/coordinator that owns the
/// infrastructure handles.  NOT stored on the slot itself.
pub struct EffectContext<'a> {
    /// Channel to send transfer requests to the [`super::LocalTransferEngine`].
    pub xfer_tx: &'a mpsc::UnboundedSender<LocalTransferRequest>,

    /// Sliding-window cache hit rate tracker.
    pub cache_stats: &'a Arc<CacheStatsTracker>,

    /// Leader handle for remote (G4) operations.
    pub leader: &'a Arc<KvbmLeader>,

    /// Block manager for host/disk pool lookups.
    pub block_manager: &'a VllmBlockManager,

    /// Per-slot pending worker transfer ops (drained by metadata build).
    pub pending_worker_ops: &'a Mutex<HashMap<SlotKey, Vec<WorkerTransferRequest>>>,

    /// Single source of truth for transfer completion signaling.
    pub transfer_signal: &'a Arc<dyn super::transfer_signal::TransferSignal>,
}

/// Result of executing a single effect.
///
/// Most effects are fire-and-forget (`Done`). Some, like `RunLocalLookup`,
/// produce a follow-up event that must be fed back into the state machine.
/// Match-reporting effects propagate their outcome so the caller (leader)
/// can observe the match status without re-inspecting the phase.
pub enum EffectResult {
    Done,
    FollowUp(SlotEvent<VllmHostBlocks, VllmDiskBlocks>),
    MatchReady { num_external_tokens: usize },
    MatchDeferred,
    MatchNone,
}

/// Execute a single side effect produced by the slot state machine reducer.
///
/// Each [`SlotEffect`] maps to one or more I/O operations.  Returns an
/// [`EffectResult`] indicating whether the effect completed (`Done`),
/// produced a follow-up event, or is a match-reporting signal.
pub fn execute_effect(
    effect: SlotEffect<VllmHostBlocks, VllmDiskBlocks>,
    slot: &mut super::VllmConnectorSlot,
    ctx: &EffectContext<'_>,
) -> anyhow::Result<EffectResult> {
    match effect {
        // ── I/O delegation ───────────────────────────────────────────

        SlotEffect::RunLocalLookup { num_computed_tokens } => {
            let block_size = slot.block_size;
            let computed_blocks = num_computed_tokens / block_size;
            let sequence_hashes: Vec<u64> = slot.sequence()
                .blocks()
                .iter()
                .skip(computed_blocks)
                .map(|tb| tb.sequence_hash())
                .collect();

            if sequence_hashes.is_empty() {
                return Ok(EffectResult::FollowUp(SlotEvent::LocalLookupCompleted {
                    host_blocks: vec![],
                    disk_blocks: vec![],
                    remote_candidates: vec![],
                }));
            }

            let host_blocks: VllmHostBlocks = ctx.block_manager.host()
                .map(|host| host.match_sequence_hashes_blocking(&sequence_hashes))
                .transpose()
                .unwrap_or_default()
                .unwrap_or_default();
            let num_host = host_blocks.len();

            let disk_blocks: VllmDiskBlocks = ctx.block_manager.disk()
                .map(|disk| disk.match_sequence_hashes_blocking(&sequence_hashes[num_host..]))
                .transpose()
                .unwrap_or_default()
                .unwrap_or_default();
            let num_disk = disk_blocks.len();

            let remote_candidates = sequence_hashes[num_host + num_disk..].to_vec();

            slot.performed_cache_lookup = true;
            slot.total_blocks_queried = sequence_hashes.len();

            tracing::info!(
                request_id = %slot.request_id(),
                num_host, num_disk,
                num_remote_candidates = remote_candidates.len(),
                "RunLocalLookup complete"
            );

            Ok(EffectResult::FollowUp(SlotEvent::LocalLookupCompleted {
                host_blocks: host_blocks.into_iter().map(|b| vec![b]).collect(),
                disk_blocks: disk_blocks.into_iter().map(|b| vec![b]).collect(),
                remote_candidates,
            }))
        }

        SlotEffect::StartRemoteLookup {
            candidates,
            host_blocks,
            disk_blocks,
        } => {
            if let Some(remote_handle) = ctx.leader.remote_handle() {
                tracing::info!(
                    request_id = %slot.request_id(),
                    num_candidates = candidates.len(),
                    num_host = host_blocks.len(),
                    num_disk = disk_blocks.len(),
                    "StartRemoteLookup: initiating G4 registry lookup"
                );

                let world_size = ctx.leader.world_size();
                let matched_hashes = super::registry_ops::match_prefix_tp_blocking(
                    &remote_handle,
                    &candidates,
                    world_size,
                );

                if matched_hashes.is_empty() {
                    tracing::debug!(
                        request_id = %slot.request_id(),
                        "StartRemoteLookup: no G4 matches, using local blocks only"
                    );
                    return Ok(EffectResult::FollowUp(SlotEvent::RemoteLookupClosed));
                }

                tracing::info!(
                    request_id = %slot.request_id(),
                    num_matched = matched_hashes.len(),
                    "StartRemoteLookup: G4 registry matched"
                );

                let matches: Vec<(u64, u64)> = matched_hashes.iter()
                    .map(|&h| (h, h))
                    .collect();

                return Ok(EffectResult::FollowUp(SlotEvent::RemoteLookupCompleted {
                    matches,
                }));
            } else {
                tracing::debug!(
                    request_id = %slot.request_id(),
                    num_candidates = candidates.len(),
                    "StartRemoteLookup: no remote handle, falling back to local blocks"
                );
                return Ok(EffectResult::FollowUp(SlotEvent::RemoteLookupClosed));
            }
        }

        SlotEffect::StartPrefetch {
            hashes,
            token_blocks: _,
            operation_id,
        } => {
            if ctx.leader.remote_handle().is_some() {
                let key = slot.slot_key();
                let block_size = slot.block_size;

                let resolved_token_blocks: Vec<TokenBlock> = hashes.iter()
                    .filter_map(|&hash| {
                        slot.sequence().blocks().iter()
                            .find(|tb| tb.sequence_hash() == hash)
                            .cloned()
                    })
                    .collect();

                let request = RemoteTransferRequest {
                    key: key.clone(),
                    request_id: slot.request_id().to_string(),
                    sequence_hashes: hashes.clone(),
                    device_block_ids: vec![],
                    host_block_ids: None,
                    operation_id,
                    block_size,
                    is_onboard: true,
                    pin_id: None,
                    token_blocks: Some(resolved_token_blocks),
                    traceparent: slot.traceparent.clone(),
                    baggage: slot.baggage.clone(),
                };

                ctx.xfer_tx.send(LocalTransferRequest::Remote(request))
                    .map_err(|e| anyhow::anyhow!("failed to send prefetch request: {}", e))?;

                tracing::info!(
                    request_id = %slot.request_id(),
                    num_hashes = hashes.len(),
                    operation_id = %operation_id,
                    "StartPrefetch: dispatched G4->host prefetch via transfer engine"
                );

                Ok(EffectResult::Done)
            } else {
                tracing::debug!(
                    request_id = %slot.request_id(),
                    num_hashes = hashes.len(),
                    "StartPrefetch: no remote handle, falling back"
                );
                return Ok(EffectResult::FollowUp(SlotEvent::PrefetchFailed));
            }
        }

        SlotEffect::EnqueueOnboardTransfer {
            host_staging,
            disk_staging,
            remote_hashes,
            dst_blocks,
            num_external_tokens,
        } => {
            let key = slot.slot_key();
            let block_size = slot.block_size;
            let mut dst_offset = 0;
            let total_host_blocks: usize = host_staging.iter().map(Vec::len).sum();
            let total_disk_blocks: usize = disk_staging.iter().map(Vec::len).sum();
            let total_remote_blocks = remote_hashes.len();
            let total_required_blocks = total_host_blocks + total_disk_blocks + total_remote_blocks;

            if total_required_blocks > dst_blocks.len() {
                tracing::warn!(
                    request_id = %slot.request_id(),
                    required = total_required_blocks,
                    available = dst_blocks.len(),
                    host = total_host_blocks,
                    disk = total_disk_blocks,
                    remote = total_remote_blocks,
                    "EnqueueOnboardTransfer: insufficient dst blocks; preempting slot"
                );
                return Ok(EffectResult::FollowUp(SlotEvent::Preempt));
            }

            // host_staging is Vec<Vec<ImmutableBlock<PinnedStorage,...>>> -- flatten
            let flat_host: Vec<_> = host_staging.into_iter().flatten().collect();
            if !flat_host.is_empty() {
                let n = std::cmp::min(flat_host.len(), dst_blocks.len().saturating_sub(dst_offset));
                if n == 0 {
                    anyhow::bail!(
                        "EnqueueOnboardTransfer: host staging present but no destination blocks remain for request {}",
                        slot.request_id()
                    );
                }
                let flat_host: Vec<_> = flat_host.into_iter().take(n).collect();
                let host_dst = dst_blocks[dst_offset..dst_offset + n].to_vec();
                dst_offset += n;

                let operation_id = uuid::Uuid::new_v4();
                let request = LocalOnboardRequest::new(
                    key.clone(),
                    Box::new(AnyImmutableBlocks::<PinnedStorage, VllmLocality, BasicMetadata>::new(flat_host)),
                    host_dst.clone(),
                    operation_id,
                    slot.traceparent.clone(),
                );
                ctx.xfer_tx.send(LocalTransferRequest::Onboard(request))
                    .map_err(|e| anyhow::anyhow!("failed to send host onboard request: {}", e))?;
                if let Ok(mut ops) = ctx.pending_worker_ops.lock() {
                    ops.entry(key.clone()).or_default().push(WorkerTransferRequest {
                        key: key.clone(),
                        uuid: operation_id,
                        transfer_type: TransferType::Load,
                        request_type: RequestType::Immediate,
                        block_ids: host_dst.clone(),
                    });
                }
                ctx.transfer_signal.register(
                    operation_id,
                    slot.request_id(),
                    TransferType::Load,
                );

                tracing::info!(
                    request_id = %slot.request_id(),
                    num_blocks = n,
                    operation_id = %operation_id,
                    "EnqueueOnboardTransfer: dispatched host->device"
                );
            }

            // disk_staging is Vec<Vec<ImmutableBlock<DiskStorage,...>>> -- flatten
            let flat_disk: Vec<_> = disk_staging.into_iter().flatten().collect();
            if !flat_disk.is_empty() {
                let n = std::cmp::min(flat_disk.len(), dst_blocks.len().saturating_sub(dst_offset));
                if n == 0 {
                    anyhow::bail!(
                        "EnqueueOnboardTransfer: disk staging present but no destination blocks remain for request {}",
                        slot.request_id()
                    );
                }
                let flat_disk: Vec<_> = flat_disk.into_iter().take(n).collect();
                let disk_dst = dst_blocks[dst_offset..dst_offset + n].to_vec();
                dst_offset += n;

                let operation_id = uuid::Uuid::new_v4();
                let request = LocalOnboardRequest::new(
                    key.clone(),
                    Box::new(AnyImmutableBlocks::<DiskStorage, VllmLocality, BasicMetadata>::new(flat_disk)),
                    disk_dst.clone(),
                    operation_id,
                    slot.traceparent.clone(),
                );
                ctx.xfer_tx.send(LocalTransferRequest::Onboard(request))
                    .map_err(|e| anyhow::anyhow!("failed to send disk onboard request: {}", e))?;
                if let Ok(mut ops) = ctx.pending_worker_ops.lock() {
                    ops.entry(key.clone()).or_default().push(WorkerTransferRequest {
                        key: key.clone(),
                        uuid: operation_id,
                        transfer_type: TransferType::Load,
                        request_type: RequestType::Immediate,
                        block_ids: disk_dst.clone(),
                    });
                }
                ctx.transfer_signal.register(
                    operation_id,
                    slot.request_id(),
                    TransferType::Load,
                );

                tracing::info!(
                    request_id = %slot.request_id(),
                    num_blocks = n,
                    operation_id = %operation_id,
                    "EnqueueOnboardTransfer: dispatched disk->device"
                );
            }

            if !remote_hashes.is_empty() {
                let n = std::cmp::min(remote_hashes.len(), dst_blocks.len().saturating_sub(dst_offset));
                if n == 0 {
                    anyhow::bail!(
                        "EnqueueOnboardTransfer: remote hashes present but no destination blocks remain for request {}",
                        slot.request_id()
                    );
                } else {
                    let remote_hashes: Vec<u64> = remote_hashes.into_iter().take(n).collect();
                    let remote_dst = dst_blocks[dst_offset..dst_offset + n].to_vec();
                    dst_offset += n;
                    let operation_id = uuid::Uuid::new_v4();

                    let remote_token_blocks: Vec<TokenBlock> = remote_hashes.iter()
                        .filter_map(|&hash| {
                            slot.sequence().blocks().iter()
                                .find(|tb| tb.sequence_hash() == hash)
                                .cloned()
                        })
                        .collect();

                    let request = RemoteTransferRequest {
                        key: key.clone(),
                        request_id: slot.request_id().to_string(),
                        sequence_hashes: remote_hashes,
                        device_block_ids: remote_dst.clone(),
                        host_block_ids: None,
                        operation_id,
                        block_size,
                        is_onboard: true,
                        pin_id: None,
                        token_blocks: Some(remote_token_blocks),
                        traceparent: slot.traceparent.clone(),
                        baggage: slot.baggage.clone(),
                    };
                    ctx.xfer_tx.send(LocalTransferRequest::Remote(request))
                        .map_err(|e| anyhow::anyhow!("failed to send remote onboard request: {}", e))?;
                    if let Ok(mut ops) = ctx.pending_worker_ops.lock() {
                        ops.entry(key.clone()).or_default().push(WorkerTransferRequest {
                            key: key.clone(),
                            uuid: operation_id,
                            transfer_type: TransferType::Load,
                            request_type: RequestType::Immediate,
                            block_ids: remote_dst.clone(),
                        });
                    }
                    ctx.transfer_signal.register(
                        operation_id,
                        slot.request_id(),
                        TransferType::Load,
                    );

                    tracing::info!(
                        request_id = %slot.request_id(),
                        num_blocks = n,
                        operation_id = %operation_id,
                        "EnqueueOnboardTransfer: dispatched remote->device"
                    );
                }
            }

            let onboarded_blocks = dst_offset;
            slot.evaluated_blocks = std::cmp::max(
                slot.evaluated_blocks,
                onboarded_blocks,
            );

            tracing::info!(
                request_id = %slot.request_id(),
                num_external_tokens,
                total_dst = dst_blocks.len(),
                actual_dispatched = onboarded_blocks,
                evaluated_blocks = slot.evaluated_blocks,
                "EnqueueOnboardTransfer: all transfers dispatched"
            );

            Ok(EffectResult::Done)
        }

        SlotEffect::EnqueueOffloadTransfer(offload_req) => {
            let key = slot.slot_key();
            let operation_id = uuid::Uuid::new_v4();
            let block_size = slot.block_size;

            // The offload covers the most-recently-evaluated range: blocks
            // [evaluated_blocks - N .. evaluated_blocks).  These are the blocks
            // that were just processed on-device and are now eligible for D2H.
            let all_blocks = slot.sequence().blocks();
            let num_blocks = offload_req.block_ids.len();
            let block_end = slot.evaluated_blocks;
            let block_start = block_end.saturating_sub(num_blocks);
            let token_blocks: Vec<TokenBlock> = if block_end <= all_blocks.len() {
                all_blocks[block_start..block_end].to_vec()
            } else {
                tracing::warn!(
                    request_id = %slot.request_id(),
                    block_start, block_end,
                    total_sequence_blocks = all_blocks.len(),
                    "EnqueueOffloadTransfer: token block range out of bounds, using empty"
                );
                vec![]
            };

            let request = LocalOffloadRequest::new(
                key,
                offload_req.block_ids.clone(),
                token_blocks,
                offload_req.priorities,
                operation_id,
                block_size,
                slot.traceparent.clone(),
                slot.baggage.clone(),
            );

            ctx.xfer_tx.send(LocalTransferRequest::Offload(request))
                .map_err(|e| anyhow::anyhow!("failed to send offload request: {}", e))?;
            if let Ok(mut ops) = ctx.pending_worker_ops.lock() {
                ops.entry(slot.slot_key()).or_default().push(WorkerTransferRequest {
                    key: slot.slot_key(),
                    uuid: operation_id,
                    transfer_type: TransferType::Store,
                    request_type: RequestType::Scheduled,
                    block_ids: offload_req.block_ids.clone(),
                });
            }
            ctx.transfer_signal.register(
                operation_id,
                slot.request_id(),
                TransferType::Store,
            );

            tracing::debug!(
                request_id = %slot.request_id(),
                operation_id = %operation_id,
                num_blocks,
                "EnqueueOffloadTransfer: dispatched D2H"
            );

            Ok(EffectResult::Done)
        }

        // ── Match reporting ──────────────────────────────────────────

        SlotEffect::MatchReady {
            num_external_tokens,
        } => {
            tracing::info!(
                request_id = %slot.request_id(),
                num_external_tokens,
                "effect: MatchReady",
            );
            Ok(EffectResult::MatchReady { num_external_tokens })
        }

        SlotEffect::MatchDeferred => {
            tracing::trace!(
                request_id = %slot.request_id(),
                "effect: MatchDeferred",
            );
            Ok(EffectResult::MatchDeferred)
        }

        SlotEffect::MatchNone => {
            tracing::debug!(
                request_id = %slot.request_id(),
                "effect: MatchNone",
            );
            Ok(EffectResult::MatchNone)
        }

        // ── Resource lifecycle ───────────────────────────────────────

        SlotEffect::ReleaseStaging => {
            tracing::debug!(
                request_id = %slot.request_id(),
                "effect: ReleaseStaging (staging already dropped by ownership transfer)",
            );
            Ok(EffectResult::Done)
        }

        // ── Cache stats ──────────────────────────────────────────────

        SlotEffect::RecordCacheStats {
            host_blocks: _,
            disk_blocks: _,
            remote_blocks: _,
            prefetched_blocks: _,
            total_queried: _,
        } => {
            // The reducer passes placeholder zeros; read real values from
            // the slot's cross-phase counters.
            let host = slot.tokens_cached_from_host / slot.block_size;
            let disk = slot.tokens_cached_from_disk / slot.block_size;
            let remote = slot.tokens_cached_from_remote / slot.block_size;
            let total = slot.total_blocks_queried;

            ctx.cache_stats.record(host, disk, remote, total);

            tracing::debug!(
                request_id = %slot.request_id(),
                host, disk, remote, total,
                "effect: RecordCacheStats (from slot counters)",
            );
            Ok(EffectResult::Done)
        }

        // ── Diagnostics ──────────────────────────────────────────────

        SlotEffect::Diag { level, message } => {
            match level {
                DiagLevel::Info => {
                    tracing::info!(request_id = %slot.request_id(), "{}", message)
                }
                DiagLevel::Warn => {
                    tracing::warn!(request_id = %slot.request_id(), "{}", message)
                }
                DiagLevel::Error => {
                    tracing::error!(request_id = %slot.request_id(), "{}", message)
                }
            }
            Ok(EffectResult::Done)
        }
    }
}

/// Execute all effects from a reducer step, in order.
pub fn execute_effects(
    effects: Vec<SlotEffect<VllmHostBlocks, VllmDiskBlocks>>,
    slot: &mut super::VllmConnectorSlot,
    ctx: &EffectContext<'_>,
) -> anyhow::Result<Vec<EffectResult>> {
    let mut results = Vec::new();
    for effect in effects {
        results.push(execute_effect(effect, slot, ctx)?);
    }
    Ok(results)
}

/// Like [`execute_effects`], but recursively drains [`EffectResult::FollowUp`] the same way as
/// [`apply_and_execute`]. Used when effects were produced outside `apply_and_execute` (e.g.
/// scheduler output already applied the reducer on the slot).
pub fn execute_effects_recursive(
    effects: Vec<SlotEffect<VllmHostBlocks, VllmDiskBlocks>>,
    slot: &mut super::VllmConnectorSlot,
    slot_ctx: &SlotContext,
    ctx: &EffectContext<'_>,
) -> anyhow::Result<Vec<EffectResult>> {
    let mut results = Vec::new();
    for effect in effects {
        let result = execute_effect(effect, slot, ctx)?;
        match result {
            EffectResult::FollowUp(follow_up_event) => {
                let nested = apply_and_execute(slot, follow_up_event, slot_ctx, ctx)?;
                results.extend(nested);
            }
            other => {
                results.push(other);
            }
        }
    }
    Ok(results)
}

/// Apply an event to a slot and execute all resulting effects.
///
/// Follow-up events (from effects like `RunLocalLookup`) are recursively
/// fed back into the state machine. Terminal results (`MatchReady`,
/// `MatchDeferred`, `MatchNone`, `Done`) are collected for the caller.
pub fn apply_and_execute(
    slot: &mut super::VllmConnectorSlot,
    event: SlotEvent<VllmHostBlocks, VllmDiskBlocks>,
    slot_ctx: &SlotContext,
    effect_ctx: &EffectContext<'_>,
) -> anyhow::Result<Vec<EffectResult>> {
    let phase = slot.take_phase();
    let (new_phase, effects) = phase.apply(event, slot_ctx);
    slot.set_phase(new_phase);

    let mut results = Vec::new();
    for effect in effects {
        let result = execute_effect(effect, slot, effect_ctx)?;
        match result {
            EffectResult::FollowUp(follow_up_event) => {
                let nested = apply_and_execute(slot, follow_up_event, slot_ctx, effect_ctx)?;
                results.extend(nested);
            }
            other => {
                results.push(other);
            }
        }
    }
    Ok(results)
}
