// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{
    collections::{HashMap, HashSet},
    sync::{Arc, OnceLock},
};

use crate::block_manager::{
    block::BlockId,
    connector::protocol::{RequestType, SlotKey, TransferType, WorkerTransferRequest},
    distributed::{KvbmLeader, vllm::is_dev_mode},
    metrics_kvbm::KvbmMetrics,
};
use serde::{Deserialize, Serialize};

use super::{ConnectorSlotManager, SlotManager, SlotState, VllmConnectorSlot};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulerOutput {
    pub new_requests: Vec<NewRequestData>,
    pub cached_requests: Vec<CachedRequestData>,
    pub num_scheduled_tokens: std::collections::HashMap<String, usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NewRequestData {
    pub request_id: String,
    pub prompt_token_ids: Vec<u32>,
    pub block_ids: Vec<BlockId>,
    pub num_computed_tokens: usize,
    pub priorities: Option<Vec<u32>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedRequestData {
    pub request_id: String,
    pub resumed_from_preemption: bool,
    pub new_token_ids: Vec<u32>,
    pub new_block_ids: Vec<BlockId>,
    pub num_computed_tokens: usize,
    pub priorities: Option<Vec<u32>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NewSlotKind {
    /// Fresh vLLM slot for normal prefill/decode scheduling. These slots must
    /// not carry any immediate onboarding load operations in the same metadata
    /// batch.
    Prefill,
    /// Slot entering async remote-KV onboarding. The worker must derive a
    /// strictly-positive number of immediate load ops for this slot from the
    /// accompanying `operations` payload before creating the scheduler epoch.
    Onboarding,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NewSlotInfo {
    pub key: SlotKey,
    pub kind: NewSlotKind,
    /// Leader-provided checksum for onboarding cardinality. The worker must
    /// derive the real immediate load-op count from `operations` and validate
    /// it against this value before creating the scheduler epoch.
    pub expected_immediate_ops: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectorMetadata {
    pub iteration: u64,
    pub new_slots: Vec<NewSlotInfo>,
    pub operations: Vec<WorkerTransferRequest>,
    #[serde(default)]
    pub loads_done: HashSet<String>,
    #[serde(default)]
    pub stores_done: HashSet<String>,
    #[serde(default)]
    pub failed: HashSet<String>,
}

impl ConnectorMetadata {
    pub fn new(iteration: u64) -> Self {
        Self {
            iteration,
            new_slots: Vec::new(),
            operations: Vec::new(),
            loads_done: HashSet::new(),
            stores_done: HashSet::new(),
            failed: HashSet::new(),
        }
    }

    fn push_slot(
        &mut self,
        key: SlotKey,
        kind: NewSlotKind,
        expected_immediate_ops: u64,
    ) {
        self.new_slots.push(NewSlotInfo {
            key,
            kind,
            expected_immediate_ops,
        });
    }

    /// Create a fresh prefill/decode slot. Prefill slots must not require any
    /// immediate onboarding load completions in the same metadata batch.
    pub fn create_prefill_slot(&mut self, key: SlotKey) {
        self.push_slot(key, NewSlotKind::Prefill, 0);
    }

    /// Create a slot that is entering worker-side onboarding. These slots
    /// must carry at least one immediate load op in `operations`.
    pub fn create_onboarding_slot(&mut self, key: SlotKey, expected_immediate_ops: u64) {
        debug_assert!(
            expected_immediate_ops > 0,
            "onboarding slots must declare at least one immediate load op"
        );
        self.push_slot(key, NewSlotKind::Onboarding, expected_immediate_ops);
    }

    pub fn add_operations(&mut self, xfer_reqs: Vec<WorkerTransferRequest>) {
        self.operations.extend(xfer_reqs);
    }

    pub fn add_operations_for_key(
        &mut self,
        key: &SlotKey,
        mut xfer_reqs: Vec<WorkerTransferRequest>,
    ) {
        for req in &mut xfer_reqs {
            req.key = key.clone();
        }
        self.operations.extend(xfer_reqs);
    }
}

pub struct KvConnectorLeaderCore {
    slot_manager: Arc<OnceLock<ConnectorSlotManager<SlotKey>>>,
    leader: Arc<KvbmLeader>,
    block_size: usize,
    inflight_requests: HashSet<String>,
    onboarding_slots: HashSet<String>,
    finishing_requests: HashSet<String>,
    iteration_counter: u64,
    kvbm_metrics: KvbmMetrics,
    /// Maps request_id -> W3C traceparent so all spans for one request share a single trace ID.
    request_traces: HashMap<String, String>,
    /// Maps request_id -> W3C baggage for propagating benchmark metadata as span attributes.
    request_baggage: HashMap<String, String>,
    request_generations: HashMap<String, u64>,
    active_slot_keys: HashMap<String, SlotKey>,
}

impl std::fmt::Debug for KvConnectorLeaderCore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KvConnectorLeaderCore")
            .field("block_size", &self.block_size)
            .field("iteration_counter", &self.iteration_counter)
            .field("inflight_requests", &self.inflight_requests.len())
            .field("active_slot_keys", &self.active_slot_keys.len())
            .finish()
    }
}

impl KvConnectorLeaderCore {
    pub fn new(
        slot_manager: Arc<OnceLock<ConnectorSlotManager<SlotKey>>>,
        leader: Arc<KvbmLeader>,
        block_size: usize,
        kvbm_metrics: KvbmMetrics,
    ) -> Self {
        Self {
            slot_manager,
            leader,
            block_size,
            inflight_requests: HashSet::new(),
            onboarding_slots: HashSet::new(),
            finishing_requests: HashSet::new(),
            iteration_counter: 0,
            kvbm_metrics,
            request_traces: HashMap::new(),
            request_baggage: HashMap::new(),
            request_generations: HashMap::new(),
            active_slot_keys: HashMap::new(),
        }
    }

    #[inline]
    pub fn slot_manager(&self) -> &ConnectorSlotManager<SlotKey> {
        self.slot_manager
            .get()
            .expect("slot_manager not initialized")
    }

    /// Get the traceparent for a request, if one was registered at create_slot time.
    pub fn request_traceparent(&self, request_id: &str) -> Option<&str> {
        self.request_traces.get(request_id).map(|s| s.as_str())
    }

    /// Override the traceparent for a request. Called from Python when the
    /// connector can provide a traceparent from the HTTP/OTEL context.
    pub fn set_request_traceparent(&mut self, request_id: String, traceparent: String) {
        self.request_traces.insert(request_id, traceparent);
    }

    /// Store W3C baggage for a request. Baggage key-value pairs are attached
    /// as attributes on KVBM spans so traces carry benchmark/client metadata.
    pub fn set_request_baggage(&mut self, request_id: String, baggage: String) {
        self.request_baggage.insert(request_id, baggage);
    }

    /// Get the baggage for a request.
    pub fn request_baggage(&self, request_id: &str) -> Option<&str> {
        self.request_baggage.get(request_id).map(|s| s.as_str())
    }

    /// Extract a specific key from the W3C baggage string for a request.
    fn baggage_value(&self, request_id: &str, key: &str) -> Option<String> {
        let baggage = self.request_baggage.get(request_id)?;
        for item in baggage.split(',') {
            if let Some((k, v)) = item.split_once('=') {
                if k.trim() == key {
                    return Some(v.trim().to_string());
                }
            }
        }
        None
    }

    fn active_slot_key(&self, request_id: &str) -> Option<SlotKey> {
        self.active_slot_keys.get(request_id).cloned()
    }

    fn allocate_slot_key(&mut self, request_id: &str) -> SlotKey {
        if let Some(key) = self.active_slot_keys.get(request_id) {
            return key.clone();
        }

        let generation = self
            .request_generations
            .get(request_id)
            .map(|generation| generation.saturating_add(1))
            .unwrap_or(0);
        self.request_generations
            .insert(request_id.to_string(), generation);

        let key = SlotKey::new(request_id.to_string(), generation);
        self.active_slot_keys
            .insert(request_id.to_string(), key.clone());
        key
    }

    /// Enter a span linked to the request's trace. Returns the guard (drop to exit).
    /// Includes `run_id` and `benchmark` from W3C baggage as span attributes
    /// so they're queryable in ClickHouse.
    fn enter_request_span(
        &self,
        request_id: &str,
        span_name: &'static str,
    ) -> tracing::span::EnteredSpan {
        if !dynamo_runtime::logging::otel_export_enabled() {
            return tracing::Span::none().entered();
        }

        let run_id = self.baggage_value(request_id, "run_id").unwrap_or_default();
        let benchmark = self.baggage_value(request_id, "benchmark").unwrap_or_default();

        let span = if let Some(tp) = self.request_traces.get(request_id) {
            let linked = dynamo_runtime::logging::make_linked_span(span_name, tp);
            let wrapper = tracing::info_span!(
                parent: &linked,
                "kvbm_ctx",
                otel.name = span_name,
                run_id = %run_id,
                benchmark = %benchmark,
            );
            wrapper
        } else {
            tracing::info_span!(
                "kvbm_op",
                otel.name = span_name,
                request_id = request_id,
                run_id = %run_id,
                benchmark = %benchmark,
            )
        };

        span.entered()
    }

    pub fn get_num_new_matched_tokens(
        &self,
        request_id: String,
        _request_num_tokens: usize,
        num_computed_tokens: usize,
    ) -> anyhow::Result<(Option<usize>, bool)> {
        let key = self
            .active_slot_key(&request_id)
            .ok_or_else(|| anyhow::anyhow!("missing active SlotKey for request {}", request_id))?;
        crate::lock_slot!(self, &key => slot);
        slot.set_request_traceparent(self.request_traceparent(&request_id).map(str::to_string));
        slot.set_request_baggage(self.request_baggage(&request_id).map(str::to_string));
        let _span = slot
            .as_any_mut()
            .downcast_mut::<VllmConnectorSlot>()
            .map(|s| s.request_poll_span().entered())
            .unwrap_or_else(|| self.enter_request_span(&request_id, "kvbm.request_poll"));
        debug_assert!(num_computed_tokens.is_multiple_of(self.block_size));

        let current_state = slot.state();
        if current_state == SlotState::SkippedPrefill {
            slot.mark_as_prefilling(self.iteration_counter)?;
            return Ok((Some(0), false));
        }
        if current_state == SlotState::SkippedDecode {
            slot.mark_as_decoding(self.iteration_counter)?;
            return Ok((Some(0), false));
        }

        if (slot.sequence().total_tokens() - num_computed_tokens) < self.block_size {
            return Ok((Some(0), false));
        }

        let vllm_slot = slot
            .as_any_mut()
            .downcast_mut::<VllmConnectorSlot>()
            .ok_or_else(|| {
                anyhow::anyhow!("expected VllmConnectorSlot for request {}", request_id)
            })?;

        let slot_ctx = super::slot_machine::SlotContext {
            block_size: self.block_size,
            remote_enabled: self.leader.remote_handle().is_some(),
            g4_xfer_fail_policy: super::slot_machine::G4FailPolicy::default(),
        };

        // Phase gate 1: If Initialized or Preempted, fire AcquireMatches.
        // The reducer transitions to AwaitingLookup and emits RunLocalLookup.
        // Effects must run through `apply_and_execute` so `RunLocalLookup` completes and
        // recursive follow-ups (`LocalLookupCompleted`, remote lookup, prefetch, …) apply;
        // otherwise the slot never leaves AwaitingLookup / early lookup phases.
        // Prefetch failed with no local staging → don't retry the lookup
        // (it would spin forever if the host pool is exhausted). Report a
        // cold miss so vLLM does a full prefill instead.
        if matches!(
            vllm_slot.phase(),
            super::RequestPhase::Preempted { recovered_from_failure: true }
        ) {
            tracing::warn!(
                request_id = %request_id,
                "get_num_new_matched_tokens: prefetch failed (pool exhaustion?); \
                 reporting cold miss for full prefill"
            );
            return Ok((Some(0), false));
        }

        if matches!(
            vllm_slot.phase(),
            super::RequestPhase::Initialized | super::RequestPhase::Preempted { .. }
        ) {
            let effect_ctx = self.slot_manager().effect_context();
            let acquire_results = super::effect_executor::apply_and_execute(
                vllm_slot,
                super::SlotEvent::AcquireMatches { num_computed_tokens },
                &slot_ctx,
                &effect_ctx,
            )?;

            tracing::info!(
                request_id = %request_id,
                phase = ?vllm_slot.phase().as_slot_state(),
                "get_num_new_matched_tokens: AcquireMatches transition"
            );

            if acquire_results.iter().any(|r| {
                matches!(
                    r,
                    super::effect_executor::EffectResult::MatchNone
                )
            }) {
                return Ok((Some(0), false));
            }
        }

        // Phase gate 2: inspect the phase after the potential AcquireMatches transition.
        match vllm_slot.phase() {
            // Async lookup or prefetch still in flight -- defer to next poll
            super::RequestPhase::LookingUp { .. } => Ok((None, false)),

            super::RequestPhase::Prefetching { prefetch, .. } => {
                let op_id = prefetch.operation_id;
                let prefetch_hashes = prefetch.sequence_hashes.clone();
                match self.slot_manager().check_prefetch_outcome(&op_id) {
                    Some(true) => {
                        let resolved_host_blocks = self
                            .slot_manager()
                            .resolve_host_blocks_by_hash(&prefetch_hashes);
                        let resolved_count = resolved_host_blocks.len();
                        let expected = prefetch_hashes.len();
                        tracing::info!(
                            request_id = %request_id,
                            operation_id = %op_id,
                            resolved_host_blocks = resolved_count,
                            expected_blocks = expected,
                            "prefetch completed; resolved {}/{} blocks from host pool → OnboardReady",
                            resolved_count, expected,
                        );
                        let effect_ctx = self.slot_manager().effect_context();
                        if resolved_count > 0 {
                            super::effect_executor::apply_and_execute(
                                vllm_slot,
                                super::SlotEvent::PrefetchReady {
                                    blocks: resolved_host_blocks.into_iter().map(|b| vec![b]).collect(),
                                },
                                &slot_ctx,
                                &effect_ctx,
                            )?;
                        } else {
                            // Host blocks not yet visible — register_blocks
                            // is async and may still be in flight. Re-insert
                            // the operation_id so the next poll retries
                            // resolution instead of falling back to a
                            // redundant remote re-read.
                            if let Ok(mut set) = self.slot_manager().prefetch_completed.lock() {
                                set.insert(op_id);
                            }
                            tracing::debug!(
                                request_id = %request_id,
                                operation_id = %op_id,
                                "prefetch completed but host blocks not yet registered; \
                                 deferring resolution to next poll"
                            );
                        }
                        Ok((None, false))
                    }
                    Some(false) => {
                        tracing::warn!(
                            request_id = %request_id,
                            operation_id = %op_id,
                            "prefetch failed; falling back"
                        );
                        let effect_ctx = self.slot_manager().effect_context();
                        super::effect_executor::apply_and_execute(
                            vllm_slot,
                            super::SlotEvent::PrefetchFailed,
                            &slot_ctx,
                            &effect_ctx,
                        )?;
                        Ok((None, false))
                    }
                    None => {
                        let timeout = super::slot_config::prefetch_timeout();
                        if prefetch.started_at.elapsed() > timeout {
                            tracing::warn!(
                                request_id = %request_id,
                                operation_id = %op_id,
                                elapsed_secs = prefetch.started_at.elapsed().as_secs(),
                                timeout_secs = timeout.as_secs(),
                                "prefetch timed out"
                            );
                            let effect_ctx = self.slot_manager().effect_context();
                            super::effect_executor::apply_and_execute(
                                vllm_slot,
                                super::SlotEvent::PrefetchTimeout,
                                &slot_ctx,
                                &effect_ctx,
                            )?;
                            Ok((None, false))
                        } else {
                            tracing::debug!(
                                target: "kvbm-g4",
                                request_id = %request_id,
                                "host prefetch pending; deferring"
                            );
                            Ok((None, false))
                        }
                    }
                }
            }

            super::RequestPhase::AwaitingLookup => Ok((None, false)),

            // External blocks staged -- fire PollMatchReport to disclose/defer
            super::RequestPhase::OnboardReady { .. } => {
                let phase = vllm_slot.take_phase();
                let (new_phase, effects) =
                    phase.apply(super::SlotEvent::PollMatchReport, &slot_ctx);
                vllm_slot.set_phase(new_phase);

                for effect in effects {
                    match effect {
                        super::SlotEffect::MatchReady { num_external_tokens } => {
                            debug_assert!(
                                (num_computed_tokens + num_external_tokens)
                                    .is_multiple_of(self.block_size)
                            );
                            self.kvbm_metrics
                                .matched_tokens
                                .inc_by(num_external_tokens as u64);
                            tracing::info!(
                                target: "kvbm-diag",
                                request_id = %request_id,
                                num_external_tokens,
                                num_computed_tokens,
                                "get_num_new_matched_tokens → OnboardReady (returning async match)"
                            );
                            return Ok((Some(num_external_tokens), true));
                        }
                        super::SlotEffect::MatchDeferred => {
                            return Ok((None, false));
                        }
                        super::SlotEffect::MatchNone => {
                            return Ok((Some(0), false));
                        }
                        _ => {}
                    }
                }
                Ok((None, false))
            }

            super::RequestPhase::Onboarding { num_external_tokens, started_at, .. } => {
                if self.slot_manager().transfer_signal.is_loads_done(&request_id) == Some(true) {
                    tracing::info!(
                        target: "kvbm-diag",
                        request_id = %request_id,
                        num_external_tokens,
                        "get_num_new_matched_tokens: H2D complete (signal), advancing Onboarding"
                    );
                    let phase = vllm_slot.take_phase();
                    let (new_phase, _effects) = phase.apply(
                        super::SlotEvent::TransferCompleted {
                            operation_id: uuid::Uuid::nil(),
                        },
                        &slot_ctx,
                    );
                    vllm_slot.set_phase(new_phase);
                    return Ok((Some(0), false));
                }

                let elapsed = started_at.elapsed();
                let timeout = super::slot_config::prefetch_timeout();
                if elapsed > timeout {
                    tracing::warn!(
                        request_id = %request_id,
                        elapsed_secs = elapsed.as_secs(),
                        timeout_secs = timeout.as_secs(),
                        num_external_tokens,
                        "onboarding timed out (H2D completion lost?); \
                         aborting cache onboard → full prefill"
                    );
                    let effect_ctx = self.slot_manager().effect_context();
                    super::effect_executor::apply_and_execute(
                        vllm_slot,
                        super::SlotEvent::Preempt,
                        &slot_ctx,
                        &effect_ctx,
                    )?;
                    return Ok((Some(0), false));
                }
                tracing::debug!(
                    target: "kvbm-diag",
                    request_id = %request_id,
                    num_external_tokens,
                    elapsed_ms = elapsed.as_millis() as u64,
                    "get_num_new_matched_tokens: request in Onboarding phase \
                     (WAITING_FOR_REMOTE_KVS, worker H2D in progress)"
                );
                Ok((Some(0), false))
            }

            _ => Ok((Some(0), false)),
        }
    }

    pub fn update_state_after_alloc(
        &mut self,
        request_id: String,
        block_ids: Vec<BlockId>,
        num_external_tokens: usize,
    ) -> anyhow::Result<()> {
        let _span = self.enter_request_span(&request_id, "kvbm.update_state_after_alloc");
        let key = self
            .active_slot_key(&request_id)
            .ok_or_else(|| anyhow::anyhow!("missing active SlotKey for request {}", request_id))?;
        crate::lock_slot!(self, &key => slot);
        slot.set_request_traceparent(self.request_traceparent(&request_id).map(str::to_string));
        slot.set_request_baggage(self.request_baggage(&request_id).map(str::to_string));
        slot.append_mutable_device_blocks(&block_ids)?;

        if num_external_tokens > 0 {
            let (phase_before_alloc, staged_external_tokens) = {
                let vllm_slot = slot
                    .as_any_mut()
                    .downcast_mut::<VllmConnectorSlot>()
                    .ok_or_else(|| {
                        anyhow::anyhow!("expected VllmConnectorSlot for request {}", request_id)
                    })?;

                if !matches!(
                    vllm_slot.phase(),
                    super::RequestPhase::OnboardReady { .. }
                ) {
                    anyhow::bail!(
                        "update_state_after_alloc: external tokens but slot not in OnboardReady \
                         phase (phase={:?}) for request {}",
                        vllm_slot.phase().as_slot_state(),
                        request_id
                    );
                }
                let staged_external_tokens = match vllm_slot.phase() {
                    super::RequestPhase::OnboardReady { num_external_tokens, .. } => {
                        *num_external_tokens
                    }
                    _ => unreachable!("validated OnboardReady phase above"),
                };
                (vllm_slot.phase().as_slot_state(), staged_external_tokens)
            };

            let expected_blocks = staged_external_tokens.div_ceil(self.block_size);
            if block_ids.len() < expected_blocks {
                tracing::warn!(
                    target: "kvbm-diag",
                    request_id = %request_id,
                    num_external_tokens_arg = num_external_tokens,
                    staged_external_tokens,
                    expected_blocks,
                    granted_blocks = block_ids.len(),
                    phase = ?phase_before_alloc,
                    "update_state_after_alloc: undersized GPU allocation for onboard; falling back to full prefill"
                );
                let slot_ctx = super::slot_machine::SlotContext {
                    block_size: self.block_size,
                    remote_enabled: self.leader.remote_handle().is_some(),
                    g4_xfer_fail_policy: super::slot_machine::G4FailPolicy::default(),
                };
                let effect_ctx = self.slot_manager().effect_context();
                let vllm_slot = slot
                    .as_any_mut()
                    .downcast_mut::<VllmConnectorSlot>()
                    .ok_or_else(|| {
                        anyhow::anyhow!("expected VllmConnectorSlot for request {}", request_id)
                    })?;
                super::effect_executor::apply_and_execute(
                    vllm_slot,
                    super::SlotEvent::Preempt,
                    &slot_ctx,
                    &effect_ctx,
                )?;
                self.onboarding_slots.remove(&request_id);
                return Ok(());
            }

            let num_computed_tokens = block_ids.len() * self.block_size - staged_external_tokens;
            slot.record_cached_device_tokens(num_computed_tokens);

            let slot_ctx = super::slot_machine::SlotContext {
                block_size: self.block_size,
                remote_enabled: self.leader.remote_handle().is_some(),
                g4_xfer_fail_policy: super::slot_machine::G4FailPolicy::default(),
            };

            tracing::info!(
                target: "kvbm-diag",
                request_id = %request_id,
                num_external_tokens_arg = num_external_tokens,
                staged_external_tokens,
                num_device_blocks = block_ids.len(),
                num_computed_tokens,
                phase = ?phase_before_alloc,
                "update_state_after_alloc → onboarding via state machine (AllocCompleted + effects)"
            );

            let effect_ctx = self.slot_manager().effect_context();
            let vllm_slot = slot
                .as_any_mut()
                .downcast_mut::<VllmConnectorSlot>()
                .ok_or_else(|| {
                    anyhow::anyhow!("expected VllmConnectorSlot for request {}", request_id)
                })?;
            vllm_slot.trigger_onboarding_execute_effects(
                staged_external_tokens,
                Some(block_ids.as_slice()),
                &slot_ctx,
                &effect_ctx,
            )?;

            self.onboarding_slots.insert(request_id);
        }

        Ok(())
    }

    pub fn build_connector_metadata(
        &mut self,
        scheduler_output: SchedulerOutput,
    ) -> anyhow::Result<Vec<u8>> {
        self.kvbm_metrics
            .scheduler_new_requests
            .set(scheduler_output.new_requests.len() as f64);
        self.kvbm_metrics
            .scheduler_cached_requests
            .set(scheduler_output.cached_requests.len() as f64);
        self.kvbm_metrics
            .scheduler_finishing
            .set(self.finishing_requests.len() as f64);
        self.kvbm_metrics
            .scheduler_onboarding
            .set(self.onboarding_slots.len() as f64);
        self.kvbm_metrics
            .scheduler_inflight
            .set(self.inflight_requests.len() as f64);

        if !self.finishing_requests.is_empty() {
            let to_clean: Vec<String> = self.finishing_requests.drain().collect();
            for request_id in &to_clean {
                if let Some(key) = self.active_slot_key(request_id)
                    && self.slot_manager().has_slot(&key)
                {
                    let _ = self.slot_manager().remove_slot(&key);
                }
                self.active_slot_keys.remove(request_id);
                self.slot_manager().remove_signal(request_id);
            }
        }

        self.iteration_counter += 1;
        let iteration = self.iteration_counter;
        let mut inflight_requests = self.inflight_requests.clone();
        let mut md = ConnectorMetadata::new(iteration);
        let onboarding_slots = std::mem::take(&mut self.onboarding_slots);

        if !onboarding_slots.is_empty() {
            tracing::info!(
                target: "kvbm-diag",
                iteration,
                num_onboarding = onboarding_slots.len(),
                onboarding_reqs = ?onboarding_slots,
                "build_connector_metadata: flushing onboarding slots"
            );
        }

        // --- Onboarding slots: drain worker ops produced by effect executor ---
        for request_id in &onboarding_slots {
            let _req_span = self.enter_request_span(request_id, "kvbm.flush_onboarding");
            let key = self.active_slot_key(request_id).ok_or_else(|| {
                anyhow::anyhow!(
                    "missing active SlotKey for onboarding request {}",
                    request_id
                )
            })?;
            {
                crate::lock_slot!(self, &key => _slot);
            }
            if let Some(pending_ops) = self.slot_manager().take_pending_worker_ops(&key) {
                let num_immediate = pending_ops
                    .iter()
                    .filter(|op| {
                        op.transfer_type == TransferType::Load
                            && op.request_type == RequestType::Immediate
                    })
                    .count() as u64;
                if num_immediate == 0 {
                    anyhow::bail!(
                        "onboarding metadata contract violated for request {}: slot is marked for onboarding but emitted zero immediate load ops",
                        request_id
                    );
                }
                md.create_onboarding_slot(key.clone(), num_immediate);
                md.add_operations_for_key(&key, pending_ops);
            } else {
                anyhow::bail!(
                    "onboarding metadata contract violated for request {}: onboarding slot has no pending worker ops",
                    request_id
                );
            }
            if !inflight_requests.remove(request_id) {
                tracing::warn!("request {request_id} not in inflight set (may have been cleared by clear_pool)");
            }
        }

        // --- New requests ---
        for new_req in &scheduler_output.new_requests {
            let request_id = &new_req.request_id;
            let already_created = md.new_slots.iter().any(|s| s.key.request_id == *request_id);
            if already_created {
                inflight_requests.remove(request_id);
                continue;
            }

            let _req_span = self.enter_request_span(request_id, "kvbm.schedule_new_request");
            inflight_requests.remove(request_id);
            let key = self.active_slot_key(request_id).ok_or_else(|| {
                anyhow::anyhow!("missing active SlotKey for new request {}", request_id)
            })?;

            {
                crate::lock_slot!(self, &key => slot);
                slot.record_start_iteration(iteration)?;

                let scheduled_tokens = *scheduler_output
                    .num_scheduled_tokens
                    .get(request_id)
                    .unwrap_or(&0);

                tracing::info!(
                    target: "kvbm-diag",
                    request_id = %request_id,
                    iteration,
                    vllm_num_computed_tokens = new_req.num_computed_tokens,
                    vllm_num_scheduled_tokens = scheduled_tokens,
                    slot_state = ?slot.state(),
                    slot_computed_tokens = slot.computed_tokens(),
                    "build_connector_metadata: new request from vLLM scheduler"
                );

                let slot_ctx = super::slot_machine::SlotContext {
                    block_size: self.block_size,
                    remote_enabled: self.leader.remote_handle().is_some(),
                    g4_xfer_fail_policy: super::slot_machine::G4FailPolicy::default(),
                };
                let effect_ctx = self.slot_manager().effect_context();
                let vllm_slot = slot.as_any_mut().downcast_mut::<VllmConnectorSlot>().ok_or_else(
                    || {
                        anyhow::anyhow!(
                            "expected VllmConnectorSlot for new request {}",
                            request_id
                        )
                    },
                )?;
                vllm_slot.apply_scheduler_output_execute_effects(
                    &[],
                    &new_req.block_ids,
                    new_req.num_computed_tokens,
                    scheduled_tokens,
                    new_req.priorities.as_deref(),
                    &slot_ctx,
                    &effect_ctx,
                )?;
            }
            if let Some(pending_ops) = self.slot_manager().take_pending_worker_ops(&key) {
                let num_immediate_loads = pending_ops
                    .iter()
                    .filter(|op| {
                        op.transfer_type == TransferType::Load
                            && op.request_type == RequestType::Immediate
                    })
                    .count() as u64;
                if num_immediate_loads > 0 {
                    anyhow::bail!(
                        "prefill metadata contract violated for request {}: prefill slot emitted {} immediate load ops",
                        request_id,
                        num_immediate_loads
                    );
                }
                md.create_prefill_slot(key.clone());
                md.add_operations_for_key(&key, pending_ops);
            } else {
                md.create_prefill_slot(key.clone());
            }
        }

        // --- Cached requests ---
        for cached_req in &scheduler_output.cached_requests {
            let request_id = &cached_req.request_id;

            if cached_req.resumed_from_preemption {
                let key = self.active_slot_key(request_id).ok_or_else(|| {
                    anyhow::anyhow!("missing active SlotKey for cached request {}", request_id)
                })?;
                let shared_slot = self.slot_manager().get_slot(&key)?;
                let mut slot = shared_slot
                    .lock()
                    .map_err(|e| anyhow::anyhow!("Failed to lock slot: {}", e))?;
                slot.reset_after_preemption();
            }

            inflight_requests.remove(request_id);
            let key = self.active_slot_key(request_id).ok_or_else(|| {
                anyhow::anyhow!("missing active SlotKey for cached request {}", request_id)
            })?;

            {
                crate::lock_slot!(self, &key => slot);

                let scheduled_tokens = *scheduler_output
                    .num_scheduled_tokens
                    .get(request_id)
                    .unwrap_or(&0);

                let slot_ctx = super::slot_machine::SlotContext {
                    block_size: self.block_size,
                    remote_enabled: self.leader.remote_handle().is_some(),
                    g4_xfer_fail_policy: super::slot_machine::G4FailPolicy::default(),
                };
                let effect_ctx = self.slot_manager().effect_context();
                let vllm_slot = slot.as_any_mut().downcast_mut::<VllmConnectorSlot>().ok_or_else(
                    || {
                        anyhow::anyhow!(
                            "expected VllmConnectorSlot for cached request {}",
                            request_id
                        )
                    },
                )?;

                vllm_slot.apply_scheduler_output_execute_effects(
                    &cached_req.new_token_ids,
                    &cached_req.new_block_ids,
                    cached_req.num_computed_tokens,
                    scheduled_tokens,
                    cached_req.priorities.as_deref(),
                    &slot_ctx,
                    &effect_ctx,
                )?;

                tracing::debug!(
                    request_id = %key.request_id,
                    phase = ?vllm_slot.phase().as_slot_state(),
                    "build_connector_metadata: cached request phase after scheduler output"
                );
            }
            if let Some(pending_ops) = self.slot_manager().take_pending_worker_ops(&key) {
                md.add_operations_for_key(&key, pending_ops);
            }
        }

        // --- Unscheduled requests: mark as skipped ---
        for unscheduled_req in &inflight_requests {
            let key = self.active_slot_key(unscheduled_req).ok_or_else(|| {
                anyhow::anyhow!(
                    "missing active SlotKey for unscheduled request {}",
                    unscheduled_req
                )
            })?;
            crate::lock_slot!(self, &key => slot_guard);
            let slot = slot_guard
                .as_any_mut()
                .downcast_mut::<VllmConnectorSlot>()
                .ok_or_else(|| anyhow::anyhow!("Expected VllmConnectorSlot, got different type"))?;
            slot.mark_as_skipped()?;
        }

        let signal = &self.slot_manager().transfer_signal;
        for request_id in self.active_slot_keys.keys() {
            if signal.is_loads_done(request_id) == Some(true) {
                md.loads_done.insert(request_id.clone());
            }
            if signal.is_stores_done(request_id) == Some(true) {
                md.stores_done.insert(request_id.clone());
            }
            if signal.has_failed(request_id) {
                md.failed.insert(request_id.clone());
            }
        }

        serde_json::to_vec(&md)
            .map_err(|e| anyhow::anyhow!("Failed to serialize connector metadata: {}", e))
    }

    pub fn request_finished(
        &mut self,
        request_id: String,
        block_ids: Vec<BlockId>,
    ) -> anyhow::Result<bool> {
        self.onboarding_slots.remove(&request_id);
        self.request_traces.remove(&request_id);
        self.request_baggage.remove(&request_id);

        let Some(key) = self.active_slot_key(&request_id) else {
            tracing::warn!(
                "request_finished called for request_id: {request_id} but no active slot key found"
            );
            self.inflight_requests.remove(&request_id);
            self.active_slot_keys.remove(&request_id);
            self.slot_manager().remove_signal(&request_id);
            return Ok(false);
        };
        if !self.slot_manager().has_slot(&key) {
            tracing::warn!(
                "request_finished called for request_id: {request_id} but slot is not found"
            );
            self.inflight_requests.remove(&request_id);
            self.active_slot_keys.remove(&request_id);
            self.slot_manager().remove_signal(&request_id);
            return Ok(false);
        }

        let stores_done = self.slot_manager().transfer_signal
            .is_stores_done(&request_id)
            .unwrap_or(true);

        let final_state = {
            crate::lock_slot!(self, &key => slot);
            if matches!(slot.state(), SlotState::Onboarding(_))
                && let Some(vllm_slot) = slot.as_any_mut().downcast_mut::<VllmConnectorSlot>()
            {
                vllm_slot.discard_pending_operations();
            }

            slot.mark_as_finished(self.iteration_counter)?;
            self.inflight_requests.remove(&request_id);

            let state = slot.state();
            tracing::info!(
                request_id = %request_id,
                num_block_ids = block_ids.len(),
                block_ids = ?block_ids,
                phase = ?state,
                stores_done,
                "request_finished: vLLM signaled request completion"
            );
            state
        };

        match final_state {
            SlotState::Finished => {
                self.active_slot_keys.remove(&request_id);
                self.slot_manager().remove_slot(&key)?;
                self.slot_manager().remove_signal(&request_id);
            }
            SlotState::Finishing => {
                self.finishing_requests.insert(request_id);
            }
            _ => {
                self.active_slot_keys.remove(&request_id);
                self.slot_manager().remove_slot(&key)?;
                self.slot_manager().remove_signal(&request_id);
            }
        }

        Ok(!stores_done)
    }

    pub fn has_slot(&self, request_id: &str) -> bool {
        self.active_slot_key(request_id)
            .is_some_and(|key| self.slot_manager().has_slot(&key))
    }

    pub fn create_slot(
        &mut self,
        request_id: String,
        salt_hash: u64,
        tokens: Vec<u32>,
    ) -> anyhow::Result<()> {
        let key = self.allocate_slot_key(&request_id);
        self.slot_manager().create_slot(&key, tokens, salt_hash)?;
        self.inflight_requests.insert(request_id.clone());

        if !self.request_traces.contains_key(&request_id) {
            let root_span = tracing::info_span!(
                "kvbm_request",
                otel.name = "kvbm.request",
                request_id = %request_id,
            );
            let _guard = root_span.entered();
            if let Some(ctx) = dynamo_runtime::logging::get_distributed_tracing_context() {
                self.request_traces
                    .insert(request_id.clone(), ctx.create_traceparent());
            }
        }
        crate::lock_slot!(self, &key => slot);
        slot.set_request_traceparent(self.request_traceparent(&request_id).map(str::to_string));
        slot.set_request_baggage(self.request_baggage(&request_id).map(str::to_string));
        slot.set_generation(key.generation);
        Ok(())
    }

    pub fn clear_pool(&mut self, pool: String) -> anyhow::Result<()> {
        if !is_dev_mode() {
            anyhow::bail!(
                "clear_pool called but KVBM_DEV_MODE is not enabled. \
                 Set KVBM_DEV_MODE=TRUE to allow destructive pool operations."
            );
        }
        self.inflight_requests.clear();
        self.onboarding_slots.clear();
        self.finishing_requests.clear();
        self.request_generations.clear();
        self.active_slot_keys.clear();
        self.slot_manager().clear_signal();
        self.slot_manager().clear_pool(&pool)?;
        Ok(())
    }

    pub fn get_pool_status(&self) -> std::collections::HashMap<String, std::collections::HashMap<String, u64>> {
        self.slot_manager().get_pool_status()
    }
}

impl KvConnectorLeaderCore {
    /// Adapter: handle preemptions via the state machine.
    /// Called by vLLM v0.18.0 `handle_preemptions` hook.
    pub fn handle_preemptions_via_machine(&self, preempted_req_ids: &[String]) {
        let slot_ctx = super::slot_machine::SlotContext {
            block_size: self.block_size,
            remote_enabled: self.leader.remote_handle().is_some(),
            g4_xfer_fail_policy: super::slot_machine::G4FailPolicy::default(),
        };
        for req_id in preempted_req_ids {
            let slot_key = match self.active_slot_keys.get(req_id) {
                Some(key) => key.clone(),
                None => continue,
            };
            let slot_arc = match self.slot_manager().get_slot(&slot_key) {
                Ok(arc) => arc,
                Err(_) => continue,
            };
            let mut slot = match slot_arc.lock() {
                Ok(guard) => guard,
                Err(e) => {
                    tracing::error!(
                        request_id = %req_id,
                        error = %e,
                        "slot mutex poisoned in handle_preemptions_via_machine"
                    );
                    continue;
                }
            };
            if let Some(vllm_slot) =
                slot.as_any_mut().downcast_mut::<VllmConnectorSlot>()
            {
                let phase = vllm_slot.take_phase();
                let (new_phase, effects) =
                    phase.apply(super::SlotEvent::Preempt, &slot_ctx);
                vllm_slot.set_phase(new_phase);

                tracing::info!(
                    request_id = %req_id,
                    phase = ?vllm_slot.phase().as_slot_state(),
                    "handle_preemptions_via_machine: state machine transition"
                );

                for effect in effects {
                    tracing::debug!(
                        request_id = %req_id,
                        ?effect,
                        "preemption effect"
                    );
                }
            }
            self.slot_manager().remove_signal(req_id);
        }
    }

}
