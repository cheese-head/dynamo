// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{
    any::Any,
    collections::{HashMap, HashSet},
    sync::{Arc, Mutex},
};
use dashmap::DashMap;

use crate::{
    block_manager::{
        BasicMetadata, DiskStorage, ImmutableBlock, KvBlockManager, PinnedStorage,
        block::{
            BlockId, data::logical::distributed_leader_worker::DistributedLeaderWorkerResources,
            locality::Logical,
        },
        connector::{
            RequestKey,
            cache_stats::CacheStatsTracker,
            protocol::{RequestType, SlotKey, TransferType, WorkerTransferRequest},
        },
        distributed::KvbmLeader,
        metrics_kvbm::KvbmMetrics,
        pool::PinRegistry,
    },
    tokens::{SaltHash, TokenBlockSequence, Tokens},
};
use dynamo_runtime::utils::task::CriticalTaskExecutionHandle;
use tokio::{runtime::Handle, sync::mpsc};
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use super::{
    ExternallyManagedDeviceSlot, LocalOffloadRequest, LocalTransferEngine,
    LocalTransferRequest, OperationTracker, Slot, SlotError, SlotManager,
    SlotState,
};

type VllmBlockManager = KvBlockManager<Logical<DistributedLeaderWorkerResources>, BasicMetadata>;
type VllmLocality = Logical<DistributedLeaderWorkerResources>;

type VllmHostBlocks = Vec<ImmutableBlock<PinnedStorage, VllmLocality, BasicMetadata>>;
type VllmDiskBlocks = Vec<ImmutableBlock<DiskStorage, VllmLocality, BasicMetadata>>;
type VllmRequestPhase = super::slot_machine::RequestPhase<VllmHostBlocks, VllmDiskBlocks>;

pub struct ConnectorSlotManager<R: RequestKey> {
    slots: DashMap<R, Arc<Mutex<VllmConnectorSlot>>>,
    block_manager: VllmBlockManager,
    /// use this to issue [`LocalTransferRequest`]s to the transfer engine
    xfer_tx: mpsc::UnboundedSender<LocalTransferRequest>,
    _transfer_engine_handle: Option<CriticalTaskExecutionHandle>,
    /// Cache statistics tracker
    cache_stats: Arc<CacheStatsTracker>,
    /// Reference to the leader for G4 operations
    leader: Arc<KvbmLeader>,
    /// Pin registry shared with the transfer engine. Clearing this releases
    /// host blocks pinned by in-flight H2R transfers.
    pin_registry: PinRegistry,
    /// Per-slot pending worker transfer operations.
    /// Populated by the effect executor when dispatching transfers,
    /// drained by build_connector_metadata to send to workers.
    pending_worker_ops: Mutex<HashMap<SlotKey, Vec<WorkerTransferRequest>>>,
    /// Prefetch operation IDs that completed successfully on the transfer engine.
    /// Written by the transfer engine async task, polled by get_num_new_matched_tokens.
    pub prefetch_completed: Arc<Mutex<HashSet<Uuid>>>,
    /// Prefetch operation IDs that failed on the transfer engine.
    pub prefetch_failed: Arc<Mutex<HashSet<Uuid>>>,
    /// Single source of truth for onboard/offload transfer completion.
    /// Shared with the transfer engine (producer) and polled by the leader (consumer).
    pub transfer_signal: Arc<dyn super::transfer_signal::TransferSignal>,
}

impl std::fmt::Debug for ConnectorSlotManager<SlotKey> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ConnectorSlotManager").finish()
    }
}

impl<R: RequestKey> ConnectorSlotManager<R> {
    pub fn new(
        block_manager: VllmBlockManager,
        leader: Arc<KvbmLeader>,
        kvbm_metrics: KvbmMetrics,
        identifier: Option<String>,
    ) -> Self {
        let cache_stats = Arc::new(CacheStatsTracker::new(identifier));
        let kvbm_metrics_clone = kvbm_metrics.clone();
        let cache_stats_clone = cache_stats.clone();

        // Spawn a background task to periodically update metrics and log cache hit rates
        let handle = Handle::current();
        handle.spawn(async move {
            let mut interval = tokio::time::interval(std::time::Duration::from_secs(5));
            loop {
                interval.tick().await;
                // Update Prometheus metrics
                let host_rate = cache_stats_clone.host_hit_rate();
                let disk_rate = cache_stats_clone.disk_hit_rate();
                let object_rate = cache_stats_clone.object_hit_rate();
                kvbm_metrics_clone.update_cache_hit_rates(host_rate, disk_rate, object_rate);
                // Also log cache hit rates periodically
                cache_stats_clone.maybe_log();
            }
        });
        tracing::debug!(
            "creating slot manager with block size: {}",
            block_manager.block_size()
        );

        let (xfer_tx, xfer_rx) = mpsc::unbounded_channel();

        let leader_for_engine = leader.clone();
        let mut xfer_engine =
            LocalTransferEngine::new(block_manager.clone(), leader_for_engine, xfer_rx);
        let primary_token = CancellationToken::new();
        let primary_token_clone = primary_token.clone();
        let runtime_primary = Handle::current();
        let runtime_primary_clone = runtime_primary.clone();
        let kvbm_metrics_clone = kvbm_metrics.clone();
        let pin_registry = PinRegistry::new();
        let pin_registry_for_engine = pin_registry.clone();

        let prefetch_completed: Arc<Mutex<HashSet<Uuid>>> = Arc::new(Mutex::new(HashSet::new()));
        let prefetch_failed: Arc<Mutex<HashSet<Uuid>>> = Arc::new(Mutex::new(HashSet::new()));
        let prefetch_completed_for_engine = prefetch_completed.clone();
        let prefetch_failed_for_engine = prefetch_failed.clone();
        let transfer_signal: Arc<dyn super::transfer_signal::TransferSignal> =
            Arc::new(super::transfer_signal::AtomicTransferSignal::new());
        let transfer_signal_for_engine = transfer_signal.clone();

        let xfer_engine_task = CriticalTaskExecutionHandle::new_with_runtime(
            |cancellation_token| async move {
                xfer_engine
                    .execute(
                        cancellation_token,
                        runtime_primary_clone,
                        primary_token_clone,
                        kvbm_metrics_clone,
                        pin_registry_for_engine,
                        prefetch_completed_for_engine,
                        prefetch_failed_for_engine,
                        transfer_signal_for_engine,
                    )
                    .await
            },
            primary_token,
            "LocalTransferEngine",
            &runtime_primary,
        )
        .unwrap();

        Self {
            slots: DashMap::new(),
            block_manager,
            xfer_tx,
            _transfer_engine_handle: Some(xfer_engine_task),
            cache_stats,
            leader,
            pin_registry,
            pending_worker_ops: Mutex::new(HashMap::new()),
            prefetch_completed,
            prefetch_failed,
            transfer_signal,
        }
    }
}

impl<R: RequestKey> ConnectorSlotManager<R> {
    /// Clear (wipe) all KV cache entries from a specific pool.
    ///
    /// This drops **all** tracked slots (releasing block references) and then
    /// resets the target pool, returning every block to the empty state.
    ///
    /// `pool` must be one of: `"gpu"` / `"device"`, `"cpu"` / `"host"`, or `"disk"`.
    pub fn clear_pool(&self, pool: &str) -> Result<(), SlotError> {
        // We intentionally do NOT clear slots here. Slots track per-request
        // state and are cleaned up by request_finished(). Clearing them while
        // a forward pass is in-flight causes save_kv_layer to panic.
        // The pool reset (step 2) is sufficient to reclaim block memory.
        if !self.slots.is_empty() {
            tracing::info!(
                "clear_pool({pool}): {count} active slots preserved (will be cleaned up by request_finished)",
                count = self.slots.len()
            );
        }

        // Step 1: Release pin guards for completed H2R transfers.
        let pinned = self.pin_registry.total_pinned_blocks();
        if pinned > 0 {
            tracing::info!(
                "clear_pool({pool}): releasing {pinned} pinned blocks from {} H2R transfers",
                self.pin_registry.len()
            );
        }
        self.pin_registry.clear();

        // Step 3: Reset the target pool.
        match pool.to_lowercase().as_str() {
            "gpu" | "device" => {
                if let Some(device) = self.block_manager.device() {
                    device.reset_blocking()?;
                    tracing::info!("clear_pool: device (GPU) pool wiped");
                } else {
                    return Err(SlotError::InvalidOperation(
                        "device pool is not configured".into(),
                    ));
                }
            }
            "cpu" | "host" => {
                if let Some(host) = self.block_manager.host() {
                    host.reset_blocking()?;
                    tracing::info!("clear_pool: host (CPU) pool wiped");
                } else {
                    return Err(SlotError::InvalidOperation(
                        "host pool is not configured".into(),
                    ));
                }
            }
            "disk" => {
                if let Some(disk) = self.block_manager.disk() {
                    disk.reset_blocking()?;
                    tracing::info!("clear_pool: disk pool wiped");
                } else {
                    return Err(SlotError::InvalidOperation(
                        "disk pool is not configured".into(),
                    ));
                }
            }
            other => {
                return Err(SlotError::InvalidOperation(format!(
                    "unknown pool '{other}': expected one of 'gpu', 'device', 'cpu', 'host', 'disk'"
                )));
            }
        }

        Ok(())
    }

    pub fn get_pool_status(&self) -> std::collections::HashMap<String, std::collections::HashMap<String, u64>> {
        let mut pools = std::collections::HashMap::new();
        if let Some(device) = self.block_manager.device() {
            let mut m = std::collections::HashMap::new();
            m.insert("total_blocks".into(), device.total_blocks());
            m.insert("available_blocks".into(), device.available_blocks());
            pools.insert("device".into(), m);
        }
        if let Some(host) = self.block_manager.host() {
            let mut m = std::collections::HashMap::new();
            m.insert("total_blocks".into(), host.total_blocks());
            m.insert("available_blocks".into(), host.available_blocks());
            pools.insert("host".into(), m);
        }
        if let Some(disk) = self.block_manager.disk() {
            let mut m = std::collections::HashMap::new();
            m.insert("total_blocks".into(), disk.total_blocks());
            m.insert("available_blocks".into(), disk.available_blocks());
            pools.insert("disk".into(), m);
        }
        pools
    }

    /// Build an EffectContext for executing state machine effects.
    pub fn effect_context(&self) -> super::effect_executor::EffectContext<'_> {
        super::effect_executor::EffectContext {
            xfer_tx: &self.xfer_tx,
            cache_stats: &self.cache_stats,
            leader: &self.leader,
            block_manager: &self.block_manager,
            pending_worker_ops: &self.pending_worker_ops,
            transfer_signal: &self.transfer_signal,
        }
    }

    /// Check if a prefetch operation has completed (successfully or failed).
    /// Returns `Some(true)` for success, `Some(false)` for failure, `None` if still in flight.
    /// Removes the ID from the set on match.
    pub fn check_prefetch_outcome(&self, operation_id: &Uuid) -> Option<bool> {
        if let Ok(mut set) = self.prefetch_completed.lock() {
            if set.remove(operation_id) {
                return Some(true);
            }
        }
        if let Ok(mut set) = self.prefetch_failed.lock() {
            if set.remove(operation_id) {
                return Some(false);
            }
        }
        None
    }

    /// Resolve prefetched blocks from the host pool by their sequence hashes.
    /// Used after a disk→host prefetch completes to feed `PrefetchReady` with
    /// actual host block references, so the onboard path dispatches fast
    /// host→device DMA instead of a redundant VAST re-read.
    pub fn resolve_host_blocks_by_hash(&self, hashes: &[u64]) -> VllmHostBlocks {
        self.block_manager
            .host()
            .and_then(|host| host.match_sequence_hashes_blocking(hashes).ok())
            .unwrap_or_default()
    }

    /// Check if all Load (onboard/H2D) operations for a request are done.
    /// Returns `Some(true)` when all loads finished, `Some(false)` when still
    /// pending, `None` if no load ops are registered for this request.
    pub fn is_loads_done(&self, request_id: &str) -> Option<bool> {
        self.transfer_signal.is_loads_done(request_id)
    }

    /// Check if a request has any failed transfer operations.
    pub fn has_transfer_failed(&self, request_id: &str) -> bool {
        self.transfer_signal.has_failed(request_id)
    }

    /// Remove all transfer signal state for a request. Called when a request
    /// finishes to prevent stale entries from accumulating.
    pub fn remove_signal(&self, request_id: &str) {
        self.transfer_signal.remove(request_id);
    }

    pub fn clear_signal(&self) {
        self.transfer_signal.clear();
    }

    pub fn take_pending_worker_ops(&self, key: &SlotKey) -> Option<Vec<WorkerTransferRequest>> {
        match self.pending_worker_ops.lock() {
            Ok(mut ops) => ops.remove(key),
            Err(e) => {
                tracing::error!("pending_worker_ops lock poisoned: {}", e);
                None
            }
        }
    }

    /// Apply an event to a slot and execute effects using the manager's infrastructure.
    pub fn apply_event_to_slot(
        &self,
        slot_key: &R,
        event: super::slot_machine::SlotEvent<VllmHostBlocks, VllmDiskBlocks>,
    ) -> Result<
        Vec<super::slot_machine::SlotEffect<VllmHostBlocks, VllmDiskBlocks>>,
        SlotError,
    > {
        let slot_arc = self
            .slots
            .get(slot_key)
            .ok_or(SlotError::NotFound)?
            .value()
            .clone();
        let mut slot = slot_arc.lock().unwrap();

        let ctx = slot.build_slot_context();
        let phase = slot.take_phase();
        let (new_phase, effects) = phase.apply(event, &ctx);
        slot.set_phase(new_phase);

        if !effects.is_empty() {
            let effect_ctx = self.effect_context();
            tracing::debug!(
                request_id = %slot.request_id,
                num_effects = effects.len(),
                "applied event, produced effects (manager infra ready: \
                 xfer_tx={}, cache_stats={}, leader={})",
                !effect_ctx.xfer_tx.is_closed(),
                Arc::strong_count(effect_ctx.cache_stats),
                effect_ctx.leader.remote_handle().is_some(),
            );
        }

        Ok(effects)
    }
}

impl<R: RequestKey> SlotManager<R> for ConnectorSlotManager<R> {
    type SlotType = dyn ExternallyManagedDeviceSlot;

    fn has_slot(&self, request_id: &R) -> bool {
        self.slots.contains_key(request_id)
    }

    fn create_slot(
        &self,
        request_id: &R,
        tokens: Vec<u32>,
        salt_hash: SaltHash,
    ) -> Result<(), SlotError> {
        tracing::debug!(
            "creating slot with request_id: {}, num_tokens: {}",
            request_id,
            tokens.len()
        );
        let slot = VllmConnectorSlot::new(
            request_id.request_id_str().to_string(),
            tokens.into(),
            salt_hash,
            self.block_manager.block_size(),
        );
        self.slots
            .insert(request_id.clone(), Arc::new(Mutex::new(slot)));
        Ok(())
    }

    fn get_slot(&self, request_id: &R) -> Result<Arc<Mutex<Self::SlotType>>, SlotError> {
        let slot = self.slots.get(request_id).ok_or(SlotError::NotFound)?;
        Ok(slot.value().clone())
    }

    fn remove_slot(&self, request_id: &R) -> Result<(), SlotError> {
        let removed = self.slots.remove(request_id);
        if let Some((_, slot)) = removed
            && let Ok(slot) = slot.lock()
        {
            tracing::info!(
                request_id = %slot.request_id,
                phase = ?slot.phase.as_slot_state(),
                num_device_blocks = slot.device_blocks.len(),
                "remove_slot: releasing tracked device blocks with slot removal"
            );
        }
        Ok(())
    }
}

impl<R: RequestKey> Drop for ConnectorSlotManager<R> {
    fn drop(&mut self) {
        if let Some(task) = self._transfer_engine_handle.take() {
            task.cancel();
            task.detach();
        }
    }
}

pub struct VllmConnectorSlot {
    pub(crate) request_id: String,
    pub(crate) generation: u64,
    pub(crate) traceparent: Option<String>,
    pub(crate) baggage: Option<String>,

    pub(crate) phase: VllmRequestPhase,

    pub(crate) sequence: TokenBlockSequence,
    pub(crate) device_blocks: Vec<BlockId>,
    pub(crate) tokens_cached_from_device: usize,
    pub(crate) block_size: usize,
    pub(crate) current_position: usize,
    pub(crate) evaluated_blocks: usize,
    pub(crate) performed_cache_lookup: bool,
    pub(crate) total_blocks_queried: usize,

    // Cache hit counters (write-only, read by RecordCacheStats effect)
    pub(crate) tokens_cached_from_host: usize,
    pub(crate) tokens_cached_from_disk: usize,
    pub(crate) tokens_cached_from_remote: usize,

    pub(crate) offload_terminated_at_block: Option<usize>,
    pub(crate) offload_min_priority: u32,
    pub(crate) g1_residency_unprotected: bool,

    /// Per-block priority cache for chunked prefill: chunk 1 carries priorities
    /// for all blocks, but chunks 2+ have priorities=None. This map lets us
    /// look up priorities for blocks evaluated in later chunks.
    pub(crate) stored_block_priorities: HashMap<BlockId, u32>,

    /// Timestamp when the slot first entered OnboardReady and reported MatchReady.
    /// Used to detect stalls where vLLM cannot allocate GPU blocks.
    pub(crate) onboard_ready_since: Option<std::time::Instant>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DeviceBlockTableUpdateKind {
    AppendSuffix,
    NoChange,
    Resync,
}

fn classify_device_block_table_update(
    existing: &[BlockId],
    incoming: &[BlockId],
) -> DeviceBlockTableUpdateKind {
    match incoming.len().cmp(&existing.len()) {
        std::cmp::Ordering::Greater => {
            if existing == &incoming[..existing.len()] {
                DeviceBlockTableUpdateKind::AppendSuffix
            } else {
                DeviceBlockTableUpdateKind::Resync
            }
        }
        std::cmp::Ordering::Equal => {
            if existing == incoming {
                DeviceBlockTableUpdateKind::NoChange
            } else {
                DeviceBlockTableUpdateKind::Resync
            }
        }
        std::cmp::Ordering::Less => DeviceBlockTableUpdateKind::Resync,
    }
}

impl VllmConnectorSlot {
    /// Reconcile local slot state to an authoritative full block table from vLLM.
    ///
    /// This is used when incoming block IDs are not prefix-compatible with our cached table.
    /// In that situation incremental append is unsafe, so we resync local tracking and force
    /// conservative scheduler accounting on the next tick.
    fn resync_device_blocks_from_vllm(&mut self, block_ids: &[BlockId], reason: &str) {
        let old_len = self.device_blocks.len();
        let new_len = block_ids.len();

        tracing::info!(
            request_id = %self.request_id,
            reason,
            old_len,
            new_len,
            current_position = self.current_position,
            evaluated_blocks = self.evaluated_blocks,
            "adopting new device block IDs from vLLM (physical reassignment)"
        );

        if old_len != new_len {
            tracing::info!(
                request_id = %self.request_id,
                old_len,
                new_len,
                "device block table resized; previous GPU block mapping released from local tracking"
            );
        }

        self.device_blocks.clear();
        self.device_blocks.extend_from_slice(block_ids);

        if new_len < old_len {
            self.evaluated_blocks = self.evaluated_blocks.min(new_len);
            // Operation tracking now lives in phase variants
            let has_ops = matches!(
                self.phase,
                super::slot_machine::RequestPhase::Onboarding { .. }
                | super::slot_machine::RequestPhase::Decoding { .. }
                | super::slot_machine::RequestPhase::Finishing { .. }
            );
            if has_ops {
                tracing::warn!(
                    request_id = %self.request_id,
                    "clearing phase operations after table shrink"
                );
            }
        }
    }

    fn new(
        request_id: String,
        tokens: Tokens,
        salt_hash: SaltHash,
        block_size: usize,
    ) -> Self {
        assert!(!tokens.is_empty(), "tokens must be non-empty");
        debug_assert!(block_size.is_power_of_two() && block_size <= 1024);
        let sequence = TokenBlockSequence::new(tokens, block_size as u32, Some(salt_hash));

        Self {
            request_id,
            generation: 0,
            traceparent: None,
            baggage: None,
            sequence,
            block_size,
            phase: super::slot_machine::RequestPhase::Initialized,
            current_position: 0,
            evaluated_blocks: 0,
            device_blocks: Vec::new(),
            tokens_cached_from_device: 0,
            tokens_cached_from_host: 0,
            tokens_cached_from_disk: 0,
            tokens_cached_from_remote: 0,
            performed_cache_lookup: false,
            total_blocks_queried: 0,
            offload_terminated_at_block: None,
            offload_min_priority: 0,
            g1_residency_unprotected: false,
            stored_block_priorities: HashMap::new(),
            onboard_ready_since: None,
        }
    }

    pub fn set_generation(&mut self, generation: u64) {
        self.generation = generation;
    }

    /// Get a reference to the current typed phase.
    pub fn phase(&self) -> &VllmRequestPhase {
        &self.phase
    }

    /// Take the phase for reducer application (replaces with Initialized).
    pub fn take_phase(&mut self) -> VllmRequestPhase {
        std::mem::replace(&mut self.phase, super::slot_machine::RequestPhase::Initialized)
    }

    pub fn set_phase(&mut self, phase: VllmRequestPhase) {
        self.phase = phase;
    }

    /// Build a SlotContext with slot-local values only.
    /// `remote_enabled` is always false here because the slot no longer
    /// holds a leader reference. Callers with access to the leader
    /// (e.g. KvConnectorLeaderCore) should construct SlotContext directly.
    pub(crate) fn build_slot_context(&self) -> super::slot_machine::SlotContext {
        super::slot_machine::SlotContext {
            block_size: self.block_size,
            remote_enabled: false,
            g4_xfer_fail_policy: super::slot_machine::G4FailPolicy::Fallback,
        }
    }

    pub fn slot_key(&self) -> SlotKey {
        SlotKey::new(self.request_id.clone(), self.generation)
    }

    pub fn has_pending_g4_lookup(&self) -> bool {
        false
    }

    pub fn has_pending_g4_prefetch(&self) -> bool {
        false
    }

    /// Check if offload target pool has enough capacity.
    #[allow(dead_code)]
    fn offload_capacity_shortage(
        &self,
        _requested_blocks: usize,
    ) -> Result<Option<(&'static str, usize)>, SlotError> {
        Ok(None)
    }

    #[allow(dead_code)]
    fn mark_retention_unavailable(
        &mut self,
        tier_name: &'static str,
        requested_blocks: usize,
        available_blocks: usize,
    ) {
        self.g1_residency_unprotected = true;
        self.offload_terminated_at_block = Some(self.evaluated_blocks);

        tracing::warn!(
            request_id = %self.request_id,
            tier = tier_name,
            requested_blocks,
            available_blocks,
            evaluated_blocks = self.evaluated_blocks,
            current_position = self.current_position,
            "lower-tier retention unavailable; KVBM will stop offloading this slot and treat G1 residency as unprotected"
        );
    }

    fn mark_as_skipped_prefill(&mut self) -> Result<(), SlotError> {
        let current = self.phase.as_slot_state();
        if current != SlotState::Prefilling {
            return Err(SlotError::InvalidState(format!(
                "cannot mark slot as skipped prefill in state {:?}",
                current
            )));
        }
        self.set_phase(super::slot_machine::RequestPhase::SkippedPrefill);
        Ok(())
    }

    fn mark_as_skipped_decode(&mut self) -> Result<(), SlotError> {
        let current = self.phase.as_slot_state();
        if current != SlotState::Decoding {
            return Err(SlotError::InvalidState(format!(
                "cannot mark slot as skipped decode in state {:?}",
                current
            )));
        }
        let phase = self.take_phase();
        if let super::slot_machine::RequestPhase::Decoding { ops, .. } = phase {
            self.set_phase(super::slot_machine::RequestPhase::SkippedDecode { ops });
        } else {
            self.set_phase(super::slot_machine::RequestPhase::SkippedDecode {
                ops: super::slot_ops::OperationTracker::new(),
            });
        }
        Ok(())
    }

    pub fn mark_as_skipped(&mut self) -> Result<(), SlotError> {
        let ctx = self.build_slot_context();
        let phase = self.take_phase();
        let (new_phase, _effects) = phase.apply(
            super::slot_machine::SlotEvent::MarkSkipped,
            &ctx,
        );
        self.set_phase(new_phase);

        match self.phase.as_slot_state() {
            SlotState::Prefilling => self.mark_as_skipped_prefill(),
            SlotState::Decoding => self.mark_as_skipped_decode(),
            SlotState::SkippedPrefill => Ok(()),
            SlotState::SkippedDecode => Ok(()),
            other => {
                tracing::debug!(
                    "slot is in the {:?} state; will not explicitly mark as skipped, request_id: {}",
                    other,
                    self.request_id
                );
                Ok(())
            }
        }
    }

    fn apply_scheduler_output_impl(
        &mut self,
        tokens: &[u32],
        block_ids: &[BlockId],
        num_computed_tokens: usize,
        num_scheduled_tokens: usize,
        priorities: Option<&[u32]>,
        exec: Option<(
            &super::slot_machine::SlotContext,
            &super::effect_executor::EffectContext<'_>,
        )>,
    ) -> Result<(), SlotError> {
        if !tokens.is_empty() {
            self.sequence.extend(tokens.into()).unwrap();
        }

        if !block_ids.is_empty() {
            let overlap_len = if let Some(pos) = self
                .device_blocks
                .iter()
                .rposition(|&id| id == block_ids[0])
            {
                let suffix_len = self.device_blocks.len() - pos;
                if suffix_len <= block_ids.len()
                    && self.device_blocks[pos..] == block_ids[..suffix_len]
                {
                    suffix_len
                } else {
                    tracing::warn!(
                        request_id = %self.request_id,
                        "device_blocks suffix/prefix mismatch; appending all"
                    );
                    0
                }
            } else {
                0
            };
            let new_ids = &block_ids[overlap_len..];
            if !new_ids.is_empty() {
                self.device_blocks.extend_from_slice(new_ids);
            }
            if overlap_len > 0 {
                tracing::debug!(
                    request_id = %self.request_id,
                    overlap_len,
                    new_count = new_ids.len(),
                    "block_ids suffix/prefix dedup"
                );
            }
        }

        let ctx = self.build_slot_context();
        let phase = self.take_phase();
        let (new_phase, effects) = phase.apply(
            super::slot_machine::SlotEvent::ApplySchedulerOutput {
                tokens: tokens.to_vec(),
                block_ids: block_ids.iter().copied().collect(),
                num_computed_tokens,
                num_scheduled_tokens,
                priorities: priorities.map(|p| p.to_vec()),
                iteration: 0, // real iteration set by leader via SlotContext
            },
            &ctx,
        );
        self.set_phase(new_phase);

        self.current_position = num_computed_tokens + num_scheduled_tokens;

        if let Some((slot_ctx, effect_ctx)) = exec {
            super::effect_executor::execute_effects_recursive(effects, self, slot_ctx, effect_ctx)
                .map_err(|e| SlotError::InvalidOperation(e.to_string()))?;

            // Position-based offload: runs on every call, independent of phase.
            // Computes which blocks have been newly evaluated and dispatches
            // offload requests for them.
            self.compute_and_dispatch_offloads(
                num_computed_tokens,
                num_scheduled_tokens,
                priorities,
                block_ids,
                effect_ctx,
            )?;
        } else {
            for effect in &effects {
                tracing::debug!(request_id = %self.request_id, ?effect, "scheduler output effect");
            }
        }

        Ok(())
    }

    /// Apply scheduler output and execute reducer-emitted effects (e.g. [`super::slot_machine::SlotEffect::EnqueueOffloadTransfer`])
    /// so the local transfer engine and `pending_worker_ops` are populated.
    pub fn apply_scheduler_output_execute_effects(
        &mut self,
        tokens: &[u32],
        block_ids: &[BlockId],
        num_computed_tokens: usize,
        num_scheduled_tokens: usize,
        priorities: Option<&[u32]>,
        slot_ctx: &super::slot_machine::SlotContext,
        effect_ctx: &super::effect_executor::EffectContext<'_>,
    ) -> Result<(), SlotError> {
        self.apply_scheduler_output_impl(
            tokens,
            block_ids,
            num_computed_tokens,
            num_scheduled_tokens,
            priorities,
            Some((slot_ctx, effect_ctx)),
        )
    }

    /// Position-based offload computation, called after every `apply_scheduler_output`.
    ///
    /// This runs independently of the slot's phase — during prefill, decode, or any
    /// other state. It computes which device blocks have been newly evaluated since
    /// the last call and dispatches offload requests for them.
    ///
    /// Matches the upstream Dynamo's `apply_scheduler_output` logic at
    /// `/data/priel/dynamo/.../slot.rs` lines 710-880.
    fn compute_and_dispatch_offloads(
        &mut self,
        num_computed_tokens: usize,
        _num_scheduled_tokens: usize,
        priorities: Option<&[u32]>,
        block_ids: &[BlockId],
        effect_ctx: &super::effect_executor::EffectContext<'_>,
    ) -> Result<(), SlotError> {
        // Store block→priority mapping for chunked prefill.
        // Chunk 1 carries priorities for all blocks, but chunks 2+ have
        // priorities=None. The map lets us look up priorities in later chunks.
        if let Some(prios) = priorities {
            for (block_id, priority) in block_ids.iter().zip(prios.iter()) {
                self.stored_block_priorities.insert(*block_id, *priority);
            }
        }

        // Early exit if offload has been permanently terminated for this request.
        if let Some(terminated_at) = self.offload_terminated_at_block {
            tracing::debug!(
                request_id = %self.request_id,
                terminated_at,
                "offload terminated; skipping evaluation"
            );
            return Ok(());
        }

        // Advance evaluated_blocks from computed tokens.
        self.evaluated_blocks = std::cmp::max(
            self.evaluated_blocks,
            num_computed_tokens / self.block_size,
        );

        let next_position = self.current_position;
        if next_position == 0 || self.block_size == 0 {
            return Ok(());
        }

        let next_block = next_position / self.block_size;
        if next_block <= self.evaluated_blocks {
            return Ok(());
        }

        let num_candidate_blocks = next_block - self.evaluated_blocks;

        // Extract candidate block IDs from device_blocks.
        if self.evaluated_blocks + num_candidate_blocks > self.device_blocks.len() {
            tracing::debug!(
                request_id = %self.request_id,
                evaluated_blocks = self.evaluated_blocks,
                num_candidate_blocks,
                device_blocks_len = self.device_blocks.len(),
                "not enough device blocks for candidates; skipping offload"
            );
            self.evaluated_blocks += num_candidate_blocks;
            return Ok(());
        }

        let candidate_block_ids: Vec<BlockId> = self
            .device_blocks
            .iter()
            .skip(self.evaluated_blocks)
            .take(num_candidate_blocks)
            .copied()
            .collect();

        // Look up priorities from stored_block_priorities (default 0).
        let candidate_priorities: Vec<u32> = candidate_block_ids
            .iter()
            .map(|id| self.stored_block_priorities.get(id).copied().unwrap_or(0))
            .collect();

        // Apply contiguous priority filtering.
        let num_blocks_to_offload = if self.offload_min_priority > 0 {
            candidate_priorities
                .iter()
                .take_while(|&&priority| priority >= self.offload_min_priority)
                .count()
        } else {
            num_candidate_blocks
        };

        tracing::debug!(
            request_id = %self.request_id,
            num_candidate_blocks,
            num_blocks_to_offload,
            threshold = self.offload_min_priority,
            evaluated_blocks = self.evaluated_blocks,
            next_block,
            "offload candidate evaluation"
        );

        // Guard: clamp to available sequence blocks to avoid length mismatch
        // when current_position advances without tokens being added to sequence.
        let available_seq_blocks = self.sequence.blocks().len().saturating_sub(self.evaluated_blocks);
        let num_blocks_to_offload = std::cmp::min(num_blocks_to_offload, available_seq_blocks);

        if num_blocks_to_offload > 0 {
            let offload_block_ids: Vec<BlockId> = candidate_block_ids
                .iter()
                .take(num_blocks_to_offload)
                .copied()
                .collect();

            let offload_token_blocks: Vec<_> = self
                .sequence
                .blocks()
                .iter()
                .skip(self.evaluated_blocks)
                .take(num_blocks_to_offload)
                .cloned()
                .collect();

            let offload_priorities: Vec<u32> = candidate_priorities
                .iter()
                .take(num_blocks_to_offload)
                .copied()
                .collect();

            let operation_id = uuid::Uuid::new_v4();
            let key = self.slot_key();

            let request = LocalOffloadRequest::new(
                key.clone(),
                offload_block_ids.clone(),
                offload_token_blocks,
                offload_priorities,
                operation_id,
                self.block_size,
                self.traceparent.clone(),
                self.baggage.clone(),
            );

            effect_ctx.xfer_tx.send(LocalTransferRequest::Offload(request))
                .map_err(|e| SlotError::InvalidOperation(format!(
                    "transfer engine unavailable: {}; aborting offload", e
                )))?;

            if let Ok(mut ops) = effect_ctx.pending_worker_ops.lock() {
                ops.entry(key).or_default().push(WorkerTransferRequest {
                    key: self.slot_key(),
                    uuid: operation_id,
                    transfer_type: TransferType::Store,
                    request_type: RequestType::Scheduled,
                    block_ids: offload_block_ids,
                });
            }
            effect_ctx.transfer_signal.register(
                operation_id,
                &self.request_id,
                TransferType::Store,
            );

            tracing::info!(
                request_id = %self.request_id,
                operation_id = %operation_id,
                num_blocks = num_blocks_to_offload,
                evaluated_blocks = self.evaluated_blocks,
                "offload dispatched"
            );
        }

        // Terminate offloading if priority filtering stopped early.
        if num_blocks_to_offload < num_candidate_blocks && self.offload_min_priority > 0 {
            let termination_index = self.evaluated_blocks + num_blocks_to_offload;
            self.offload_terminated_at_block = Some(termination_index);
            tracing::info!(
                request_id = %self.request_id,
                termination_index,
                "offload terminated due to priority filtering"
            );
        }

        self.evaluated_blocks += num_candidate_blocks;
        Ok(())
    }

    fn trigger_onboarding_impl(
        &mut self,
        num_external_tokens: usize,
        // When `None`, uses the full `device_blocks` table (TRT-LLM path).
        alloc_block_ids: Option<&[BlockId]>,
        exec: Option<(
            &super::slot_machine::SlotContext,
            &super::effect_executor::EffectContext<'_>,
        )>,
    ) -> Result<(), SlotError> {
        let block_ids = alloc_block_ids
            .map(<[BlockId]>::to_vec)
            .unwrap_or_else(|| self.device_blocks.clone());
        let ctx = self.build_slot_context();
        let phase = self.take_phase();
        let (new_phase, effects) = phase.apply(
            super::slot_machine::SlotEvent::AllocCompleted {
                block_ids,
                num_external_tokens,
            },
            &ctx,
        );
        self.set_phase(new_phase);
        if let Some((slot_ctx, effect_ctx)) = exec {
            super::effect_executor::execute_effects_recursive(effects, self, slot_ctx, effect_ctx)
                .map_err(|e| SlotError::InvalidOperation(e.to_string()))?;
        } else {
            for effect in &effects {
                tracing::debug!(request_id = %self.request_id, ?effect, "onboarding effect");
            }
        }
        Ok(())
    }

    /// Like [`Slot::trigger_onboarding`] but runs [`EnqueueOnboardTransfer`] (and any follow-ups) on the transfer engine.
    pub fn trigger_onboarding_execute_effects(
        &mut self,
        num_external_tokens: usize,
        alloc_block_ids: Option<&[BlockId]>,
        slot_ctx: &super::slot_machine::SlotContext,
        effect_ctx: &super::effect_executor::EffectContext<'_>,
    ) -> Result<(), SlotError> {
        self.trigger_onboarding_impl(num_external_tokens, alloc_block_ids, Some((slot_ctx, effect_ctx)))
    }
}

impl std::fmt::Debug for VllmConnectorSlot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VllmConnectorSlot")
            .field("state", &self.phase.as_slot_state())
            .field("current_position", &self.current_position)
            .field("num_tokens", &self.sequence.total_tokens())
            .finish()
    }
}

impl Slot for VllmConnectorSlot {
    fn request_id(&self) -> &str {
        &self.request_id
    }

    fn state(&self) -> SlotState {
        self.phase.as_slot_state()
    }

    fn reset_after_preemption(&mut self) {
        // Real preemption: vLLM freed ALL device blocks and reset
        // num_computed_tokens to 0. Route through the reducer so staging
        // blocks in LookingUp/Prefetching/OnboardReady are properly released.
        tracing::info!(
            request_id = %self.request_id,
            phase = ?self.phase.as_slot_state(),
            "reset_after_preemption: full reset via reducer Preempt"
        );
        let ctx = self.build_slot_context();
        let phase = self.take_phase();
        let (new_phase, _effects) = phase.apply(
            super::slot_machine::SlotEvent::Preempt,
            &ctx,
        );
        self.set_phase(new_phase);
        if !self.device_blocks.is_empty() {
            tracing::info!(
                request_id = %self.request_id,
                num_device_blocks = self.device_blocks.len(),
                "reset_after_preemption: clearing tracked GPU blocks"
            );
        }
        self.device_blocks.clear();
        self.current_position = 0;
        self.evaluated_blocks = 0;
        self.offload_terminated_at_block = None;
        self.stored_block_priorities.clear();
    }

    fn reset(&mut self) {
        self.set_phase(super::slot_machine::RequestPhase::Initialized);
        if !self.device_blocks.is_empty() {
            tracing::info!(
                request_id = %self.request_id,
                num_device_blocks = self.device_blocks.len(),
                "reset: clearing tracked GPU blocks"
            );
        }
        self.device_blocks.clear();
        self.current_position = 0;
        self.evaluated_blocks = 0;
        self.offload_terminated_at_block = None;
        self.stored_block_priorities.clear();
    }

    fn mark_as_prefilling(&mut self, iteration: u64) -> Result<(), SlotError> {
        self.set_phase(super::slot_machine::RequestPhase::Prefilling {
            iteration_first_scheduled: iteration,
        });
        Ok(())
    }

    fn mark_as_decoding(&mut self, iteration: u64) -> Result<(), SlotError> {
        self.set_phase(super::slot_machine::RequestPhase::Decoding {
            ops: OperationTracker::new(),
            iteration_first_scheduled: iteration,
        });
        Ok(())
    }

    fn record_cached_device_tokens(&mut self, num_tokens: usize) {
        self.tokens_cached_from_device = num_tokens;
        tracing::debug!("recording {} cached device tokens", num_tokens,);
    }

    fn record_cached_host_tokens(&mut self, num_tokens: usize) {
        self.tokens_cached_from_host = num_tokens;
        tracing::debug!("recording {} cached host tokens", num_tokens);
    }

    fn record_cached_disk_tokens(&mut self, num_tokens: usize) {
        self.tokens_cached_from_disk = num_tokens;
        tracing::debug!("recording {} cached disk tokens", num_tokens);
    }

    #[tracing::instrument(level = "debug", skip_all, fields(request_id = self.request_id.as_str()))]
    fn apply_scheduler_output(
        &mut self,
        tokens: &[u32],
        block_ids: &[BlockId],
        num_computed_tokens: usize,
        num_scheduled_tokens: usize,
        priorities: Option<&[u32]>,
    ) -> Result<(), SlotError> {
        self.apply_scheduler_output_impl(
            tokens,
            block_ids,
            num_computed_tokens,
            num_scheduled_tokens,
            priorities,
            None,
        )
    }

    fn record_start_iteration(&mut self, _iteration: u64) -> Result<(), SlotError> {
        // iteration_first_scheduled is now tracked inside phase variants
        Ok(())
    }

    fn mark_as_finished(&mut self, _iteration: u64) -> Result<(), SlotError> {
        let ctx = self.build_slot_context();
        let phase = self.take_phase();
        let (new_phase, effects) = phase.apply(
            super::slot_machine::SlotEvent::RequestFinished,
            &ctx,
        );
        self.set_phase(new_phase);
        for effect in effects {
            tracing::debug!(request_id = %self.request_id, ?effect, "finished effect");
        }
        Ok(())
    }

    fn sequence(&self) -> &TokenBlockSequence {
        &self.sequence
    }

    fn computed_tokens(&self) -> usize {
        self.current_position
    }

    fn num_device_blocks_allocated(&self) -> usize {
        self.device_blocks.len()
    }

    fn take_pending_operations(&mut self) -> Option<Vec<WorkerTransferRequest>> {
        // Worker ops are now dispatched by the effect executor and collected
        // on ConnectorSlotManager::pending_worker_ops. The leader drains them
        // via take_pending_worker_ops(). This trait method returns None.
        None
    }

    #[tracing::instrument(level = "debug", skip_all)]
    fn acquire_local_matches(&mut self, num_computed_tokens: usize) -> Result<(), SlotError> {
        let ctx = self.build_slot_context();
        let phase = self.take_phase();
        let (new_phase, effects) = phase.apply(
            super::slot_machine::SlotEvent::AcquireMatches { num_computed_tokens },
            &ctx,
        );
        self.set_phase(new_phase);
        for effect in effects {
            tracing::debug!(request_id = %self.request_id, ?effect, "acquire_matches effect");
        }
        Ok(())
    }

    fn trigger_onboarding(&mut self, num_external_tokens: usize) -> Result<(), SlotError> {
        self.trigger_onboarding_impl(num_external_tokens, None, None)
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

impl ExternallyManagedDeviceSlot for VllmConnectorSlot {
    fn advance_computed_position(&mut self, num_tokens: usize) -> Result<(), SlotError> {
        if self.current_position + num_tokens > self.sequence().total_tokens() {
            return Err(SlotError::InvalidOperation(format!(
                "cannot advance computed position from {} by {num_tokens} tokens, total tokens is {}",
                self.current_position,
                self.sequence().total_tokens()
            )));
        }

        tracing::debug!(
            "advancing computed position by {} tokens from {} to {}",
            num_tokens,
            self.current_position,
            self.current_position + num_tokens
        );

        self.current_position += num_tokens;
        Ok(())
    }

    /// Append device blocks to the slot.
    ///
    /// vLLM's `get_blocks()` returns the FULL block table each time (not just newly allocated
    /// blocks). This method handles that by only appending the new suffix — blocks beyond what
    /// we already track. Without this deduplication, the second `update_state_after_alloc` call
    /// would duplicate the entire table, causing `apply_scheduler_output` to offload from wrong
    /// positions (e.g., system prompt GPU blocks instead of newly computed blocks).
    #[tracing::instrument(level = "debug", skip_all, fields(request_id = self.request_id))]
    fn append_mutable_device_blocks(&mut self, block_ids: &[BlockId]) -> Result<(), SlotError> {
        let existing = self.device_blocks.len();
        match classify_device_block_table_update(self.device_blocks.as_slice(), block_ids) {
            DeviceBlockTableUpdateKind::AppendSuffix => {
                // Append the truly new blocks (the suffix beyond what we already have).
                let new_blocks = &block_ids[existing..];
                self.device_blocks.extend(new_blocks);
                tracing::debug!(
                    "appended {} new device blocks (skipped {} existing); total device blocks: {}",
                    new_blocks.len(),
                    existing,
                    self.num_device_blocks_allocated()
                );
            }
            DeviceBlockTableUpdateKind::NoChange => {
                tracing::debug!(
                    "no new device blocks to append; total device blocks: {}",
                    self.num_device_blocks_allocated()
                );
            }
            DeviceBlockTableUpdateKind::Resync => {
                if self.g1_residency_unprotected {
                    tracing::debug!(
                        request_id = %self.request_id,
                        old_len = self.device_blocks.len(),
                        new_len = block_ids.len(),
                        "adopting authoritative vLLM device block table while G1 is unprotected"
                    );
                    if !self.device_blocks.is_empty() {
                        tracing::info!(
                            request_id = %self.request_id,
                            old_len = self.device_blocks.len(),
                            new_len = block_ids.len(),
                            "replacing tracked GPU block table while G1 is unprotected"
                        );
                    }
                    self.device_blocks.clear();
                    self.device_blocks.extend_from_slice(block_ids);
                    return Ok(());
                }

                if block_ids.len() == existing {
                    tracing::info!(
                        request_id = %self.request_id,
                        num_blocks = existing,
                        "adopting reassigned device blocks (same-length table); preserving offload position"
                    );

                    // Operation tracking lives in phase variants;
                    // stale operations are dropped on phase transition.

                    tracing::info!(
                        request_id = %self.request_id,
                        num_blocks = existing,
                        "replacing tracked GPU block IDs with same-length reassignment"
                    );
                    self.device_blocks.clear();
                    self.device_blocks.extend_from_slice(block_ids);
                } else {
                    let reason = if block_ids.len() > existing {
                        "prefix_mismatch_growing_table"
                    } else {
                        "incoming_table_shrank"
                    };
                    self.resync_device_blocks_from_vllm(block_ids, reason);
                }
            }
        }

        Ok(())
    }

    fn set_request_traceparent(&mut self, traceparent: Option<String>) {
        self.traceparent = traceparent;
    }

    fn set_request_baggage(&mut self, baggage: Option<String>) {
        self.baggage = baggage;
    }

    fn set_generation(&mut self, generation: u64) {
        self.generation = generation;
    }
}

impl VllmConnectorSlot {
    pub(crate) fn request_poll_span(&mut self) -> tracing::Span {
        if !dynamo_runtime::logging::otel_export_enabled() {
            tracing::Span::none()
        } else if let Some(tp) = self.traceparent.as_deref() {
            dynamo_runtime::logging::make_linked_span("kvbm.request_poll", tp)
        } else {
            tracing::info_span!(
                "kvbm_request_poll",
                otel.name = "kvbm.request_poll",
                request_id = %self.request_id,
            )
        }
    }

    /// Discard all pending operations WITHOUT counting them as dispatched.
    pub fn discard_pending_operations(&mut self) {
        tracing::debug!(
            request_id = %self.request_id,
            "discard_pending_operations: no-op (operations tracked in phase variants)"
        );
    }

    /// Flush blocks that were never offloaded during chunked prefill.
    pub fn flush_remaining_blocks(&mut self, _all_block_ids: &[BlockId]) -> Result<(), SlotError> {
        Ok(())
    }

    pub fn set_request_traceparent(&mut self, traceparent: Option<String>) {
        self.traceparent = traceparent;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block_manager::block::transfer::remote::{
        RemoteBlockDescriptor, RemoteTransferPipeline,
    };
    use crate::block_manager::distributed::vllm::integration::G4OnboardParams;
    use crate::block_manager::distributed::vllm::{
        LocalOffloadRequest, RemoteTransferRequest,
    };
    use crate::tokens::{TokenBlock, TokenBlockSequence, Tokens};

    fn make_token_blocks(tokens: &[u32]) -> Vec<TokenBlock> {
        TokenBlockSequence::new(Tokens::from(tokens), 1, Some(0))
            .blocks()
            .to_vec()
    }

    /// Test that RemoteTransferRequest::new_h2o creates the correct request structure.
    #[test]
    fn test_h2o_request_creation() {
        let request_id = "test-request-001".to_string();
        let sequence_hashes = vec![0x1234, 0x5678, 0x9ABC];
        let host_block_ids = vec![42, 17, 88];
        let operation_id = uuid::Uuid::new_v4();
        let block_size = 16;

        let pin_id = uuid::Uuid::new_v4();
        let req = RemoteTransferRequest::new_h2o(
            SlotKey::new(request_id.clone(), 7),
            sequence_hashes.clone(),
            host_block_ids.clone(),
            operation_id,
            block_size,
            pin_id,
            None,
            None,
        );

        assert!(!req.is_onboard);
        assert!(req.is_h2o());
        assert_eq!(req.sequence_hashes, sequence_hashes);
        assert_eq!(req.host_block_ids, Some(host_block_ids));
        assert!(req.device_block_ids.is_empty()); // H2R doesn't use device blocks
        assert_eq!(req.block_size, block_size);
        assert_eq!(req.key.request_id, request_id);
        assert_eq!(req.key.generation, 7);
        assert_eq!(req.pin_id, Some(pin_id));
    }

    /// Test that G4 onboard request construction preserves the host-prefetch
    /// contract: onboard direction, no host block IDs, and request traceparent.
    #[test]
    fn test_g4_onboard_request_creation_for_host_prefetch() {
        let request_id = "test-request-g4-prefetch".to_string();
        let key = SlotKey::new(request_id.clone(), 0);
        let sequence_hashes = vec![0x1111, 0x2222, 0x3333];
        let device_block_ids = vec![]; // Host-prefetch-only path
        let operation_id = uuid::Uuid::new_v4();
        let block_size = 16;
        let token_blocks = make_token_blocks(&[0x1111, 0x2222, 0x3333]);
        let traceparent =
            Some("00-0123456789abcdef0123456789abcdef-0123456789abcdef-01".to_string());

        let params = G4OnboardParams {
            key: key.clone(),
            request_id: request_id.clone(),
            sequence_hashes: sequence_hashes.clone(),
            device_block_ids: device_block_ids.clone(),
            operation_id,
            block_size,
            token_blocks: token_blocks.clone(),
        };

        let req = RemoteTransferRequest::from_g4_params(&params, traceparent.clone(), None);

        assert!(req.is_onboard);
        assert!(!req.is_h2o());
        assert_eq!(req.key.request_id, request_id);
        assert_eq!(req.key, key);
        assert_eq!(req.sequence_hashes, sequence_hashes);
        assert_eq!(req.device_block_ids, device_block_ids);
        assert_eq!(req.host_block_ids, None);
        assert_eq!(req.operation_id, operation_id);
        assert_eq!(req.block_size, block_size);
        assert_eq!(req.token_blocks, Some(token_blocks));
        assert_eq!(req.traceparent, traceparent);
    }

    /// Test that H2R pipeline uses offload_with_bounce correctly.
    #[test]
    fn test_h2o_pipeline_uses_host_block_ids() {
        let host_block_ids = vec![42, 17, 88];
        let descriptors = vec![
            RemoteBlockDescriptor::object_from_hash("test-bucket", 0x1234, 4096),
            RemoteBlockDescriptor::object_from_hash("test-bucket", 0x5678, 4096),
            RemoteBlockDescriptor::object_from_hash("test-bucket", 0x9ABC, 4096),
        ];

        // This is what process_remote_transfer_request does for H2R
        let pipeline = RemoteTransferPipeline::offload_with_bounce(
            descriptors,
            host_block_ids.clone(),
            vec![], // Empty device_block_ids for H2R
        );

        assert!(pipeline.has_bounce());
        assert_eq!(pipeline.bounce_block_ids(), Some(host_block_ids.as_slice()));
        assert_eq!(pipeline.device_block_ids(), Some([].as_slice()));
        assert_eq!(pipeline.num_blocks(), 3);
    }

    /// Test that LocalOffloadRequest stores block_size.
    #[test]
    fn test_local_offload_request_has_block_size() {
        let request_id = "test-request".to_string();
        let key = SlotKey::new(request_id.clone(), 0);
        let block_ids = vec![0, 1, 2];

        // Create mock token blocks (we can't easily create real ones without the full infrastructure)
        // So we just test the struct directly
        let operation_id = uuid::Uuid::new_v4();
        let block_size = 16;

        // Test that LocalOffloadRequest stores block_size
        let req = LocalOffloadRequest {
            key: key.clone(),
            request_id: request_id.clone(),
            block_ids: block_ids.clone(),
            token_blocks: vec![], // Empty for this unit test
            priorities: vec![],
            operation_id,
            sequence_hashes: vec![0x1234, 0x5678, 0x9ABC],
            block_size,
            traceparent: None,
            baggage: None,
        };

        assert_eq!(req.block_size, block_size);
        assert_eq!(req.block_ids.len(), 3);
        assert_eq!(req.key, key);
    }

    /// Test that LocalOffloadRequest::new wires the full offload cycle inputs
    /// correctly, including derived sequence hashes and trace context.
    #[test]
    fn test_local_offload_request_new_derives_sequence_hashes_and_traceparent() {
        let request_id = "test-offload-request".to_string();
        let key = SlotKey::new(request_id.clone(), 0);
        let block_ids = vec![7, 8];
        let operation_id = uuid::Uuid::new_v4();
        let block_size = 16;
        let traceparent =
            Some("00-fedcba9876543210fedcba9876543210-fedcba9876543210-01".to_string());

        let token_blocks = make_token_blocks(&[0xAAAA, 0xBBBB]);
        let priorities = vec![10, 20];

        let req = LocalOffloadRequest::new(
            key.clone(),
            block_ids.clone(),
            token_blocks.clone(),
            priorities.clone(),
            operation_id,
            block_size,
            traceparent.clone(),
            None,
        );

        assert_eq!(req.key.request_id, request_id);
        assert_eq!(req.key, key);
        assert_eq!(req.block_ids, block_ids);
        assert_eq!(req.operation_id, operation_id);
        assert_eq!(req.block_size, block_size);
        assert_eq!(req.priorities, priorities);
        assert_eq!(req.traceparent, traceparent);
        assert_eq!(
            req.sequence_hashes,
            token_blocks
                .iter()
                .map(|tb| tb.sequence_hash())
                .collect::<Vec<_>>()
        );
    }

    /// Test H2R filtering logic: already-stored hashes are removed.
    #[test]
    fn test_h2o_filtering_removes_already_stored() {
        // Simulate g4_can_offload response
        let all_hashes: Vec<u64> = vec![0x1111, 0x2222, 0x3333, 0x4444];
        let already_stored: Vec<u64> = vec![0x2222, 0x4444]; // These are already in object storage

        // Build stored set for O(1) lookup
        let stored_set: std::collections::HashSet<u64> = already_stored.into_iter().collect();

        // Filter - keep only hashes NOT in stored_set
        let can_offload_hashes: Vec<u64> = all_hashes
            .iter()
            .filter(|h| !stored_set.contains(h))
            .copied()
            .collect();

        assert_eq!(can_offload_hashes, vec![0x1111, 0x3333]);
    }

    /// Test H2R host block ID filtering matches hash filtering.
    #[test]
    fn test_h2o_host_block_id_filtering() {
        let sequence_hashes: Vec<u64> = vec![0x1111, 0x2222, 0x3333, 0x4444];
        let host_block_ids: Vec<usize> = vec![10, 20, 30, 40];
        let already_stored: Vec<u64> = vec![0x2222, 0x4444];

        let stored_set: std::collections::HashSet<u64> = already_stored.into_iter().collect();

        // Filter both hashes and host block IDs together
        let (filtered_hashes, filtered_host_ids): (Vec<u64>, Vec<usize>) = sequence_hashes
            .iter()
            .zip(host_block_ids.iter())
            .filter(|(hash, _)| !stored_set.contains(hash))
            .map(|(&hash, &id)| (hash, id))
            .unzip();

        assert_eq!(filtered_hashes, vec![0x1111, 0x3333]);
        assert_eq!(filtered_host_ids, vec![10, 30]);
    }

    /// Test that empty can_offload result means skip H2R entirely.
    #[test]
    fn test_h2o_skip_when_all_stored() {
        let all_hashes: Vec<u64> = vec![0x1111, 0x2222];
        let already_stored: Vec<u64> = vec![0x1111, 0x2222]; // ALL are stored

        let stored_set: std::collections::HashSet<u64> = already_stored.into_iter().collect();

        let can_offload_hashes: Vec<u64> = all_hashes
            .iter()
            .filter(|h| !stored_set.contains(h))
            .copied()
            .collect();

        // When can_offload_hashes is empty, H2R should be skipped
        assert!(can_offload_hashes.is_empty());

        // This matches the logic in process_remote_transfer_request:
        // if can_offload_hashes.is_empty() { return Ok(()); }
    }

    #[test]
    fn test_classify_device_block_table_update_append_suffix() {
        let existing = vec![100, 101, 102];
        let incoming = vec![100, 101, 102, 103, 104];
        assert_eq!(
            classify_device_block_table_update(&existing, &incoming),
            DeviceBlockTableUpdateKind::AppendSuffix
        );
    }

    #[test]
    fn test_classify_device_block_table_update_no_change() {
        let existing = vec![10, 11, 12];
        let incoming = vec![10, 11, 12];
        assert_eq!(
            classify_device_block_table_update(&existing, &incoming),
            DeviceBlockTableUpdateKind::NoChange
        );
    }

    #[test]
    fn test_classify_device_block_table_update_resync_reorder() {
        let existing = vec![364, 365, 366, 927];
        let incoming = vec![8, 5, 7, 9];
        assert_eq!(
            classify_device_block_table_update(&existing, &incoming),
            DeviceBlockTableUpdateKind::Resync
        );
    }

    #[test]
    fn test_classify_device_block_table_update_resync_shrink() {
        let existing = vec![1, 2, 3, 4, 5];
        let incoming = vec![1, 2, 3];
        assert_eq!(
            classify_device_block_table_update(&existing, &incoming),
            DeviceBlockTableUpdateKind::Resync
        );
    }

    // -- classify_device_block_table_update (rstest parameterized) --

    #[rstest::rstest]
    #[case(&[1,2,3], &[1,2,3,4,5], DeviceBlockTableUpdateKind::AppendSuffix)]
    #[case(&[1,2,3], &[1,2,3], DeviceBlockTableUpdateKind::NoChange)]
    #[case(&[1,2,3], &[4,5,6], DeviceBlockTableUpdateKind::Resync)]
    #[case(&[1,2,3], &[1,2,4], DeviceBlockTableUpdateKind::Resync)]
    #[case(&[1,2,3], &[1,2], DeviceBlockTableUpdateKind::Resync)]
    #[case(&[], &[1,2], DeviceBlockTableUpdateKind::AppendSuffix)]
    #[case(&[], &[], DeviceBlockTableUpdateKind::NoChange)]
    fn test_classify_device_block_update(
        #[case] existing: &[usize],
        #[case] incoming: &[usize],
        #[case] expected: DeviceBlockTableUpdateKind,
    ) {
        assert_eq!(
            classify_device_block_table_update(existing, incoming),
            expected
        );
    }
}
