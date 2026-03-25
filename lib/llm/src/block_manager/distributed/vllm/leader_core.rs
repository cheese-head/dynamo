// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{
    collections::{HashMap, HashSet},
    sync::{Arc, OnceLock},
};

use crate::block_manager::{
    block::BlockId,
    connector::protocol::{SlotKey, WorkerTransferRequest},
    distributed::vllm::is_dev_mode,
    metrics_kvbm::KvbmMetrics,
};
use serde::{Deserialize, Serialize};

use super::{ConnectorSlotManager, SlotError, SlotManager, SlotState, VllmConnectorSlot};

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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NewSlotInfo {
    pub key: SlotKey,
    pub expected_immediate_ops: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectorMetadata {
    pub iteration: u64,
    pub new_slots: Vec<NewSlotInfo>,
    pub operations: Vec<WorkerTransferRequest>,
}

impl ConnectorMetadata {
    pub fn new(iteration: u64) -> Self {
        Self {
            iteration,
            new_slots: Vec::new(),
            operations: Vec::new(),
        }
    }

    pub fn create_slot_with_key(&mut self, key: SlotKey, expected_immediate_ops: u64) {
        self.new_slots.push(NewSlotInfo {
            key,
            expected_immediate_ops,
        });
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

#[derive(Debug)]
pub struct KvConnectorLeaderCore {
    slot_manager: Arc<OnceLock<ConnectorSlotManager<SlotKey>>>,
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

impl KvConnectorLeaderCore {
    pub fn new(
        slot_manager: Arc<OnceLock<ConnectorSlotManager<SlotKey>>>,
        block_size: usize,
        kvbm_metrics: KvbmMetrics,
    ) -> Self {
        Self {
            slot_manager,
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

        debug_assert!(
            slot.state() != SlotState::Prefilling && slot.state() != SlotState::Decoding,
            "slot is in the Prefilled state or Decoding; shouldn't happen"
        );

        if slot.state() == SlotState::SkippedPrefill || slot.state() == SlotState::SkippedDecode {
            match slot.state() {
                SlotState::SkippedPrefill => {
                    slot.mark_as_prefilling(self.iteration_counter)?;
                    return Ok((Some(0), false));
                }
                SlotState::SkippedDecode => {
                    slot.mark_as_decoding(self.iteration_counter)?;
                    return Ok((Some(0), false));
                }
                _ => unreachable!("slot is not in the SkippedPrefill or SkippedDecode state"),
            }
        }

        if (slot.sequence().total_tokens() - num_computed_tokens) < self.block_size {
            return Ok((Some(0), false));
        }

        slot.acquire_local_matches(num_computed_tokens)?;

        if slot
            .as_any_mut()
            .downcast_mut::<VllmConnectorSlot>()
            .map(|s| s.has_pending_g4_lookup())
            .unwrap_or(false)
        {
            return Ok((None, false));
        }

        if slot
            .as_any_mut()
            .downcast_mut::<VllmConnectorSlot>()
            .map(|s| s.has_pending_g4_prefetch())
            .unwrap_or(false)
        {
            tracing::debug!(
                target: "kvbm-g4",
                request_id = %request_id,
                "host prefetch still pending; deferring matched-token return"
            );
            return Ok((None, false));
        }

        if let SlotState::OnboardStaged(num_external_tokens) = slot.state() {
            let vllm_slot = slot
                .as_any_mut()
                .downcast_mut::<VllmConnectorSlot>()
                .ok_or_else(|| {
                    anyhow::anyhow!("expected VllmConnectorSlot for request {}", request_id)
                })?;
            let Some(num_external_tokens) = vllm_slot.disclose_staged_match_to_scheduler() else {
                if vllm_slot.staged_match_report().is_none() {
                    tracing::debug!(
                        target: "kvbm-diag",
                        request_id = %request_id,
                        num_external_tokens,
                        "get_num_new_matched_tokens → OnboardStaged without a pending report; waiting for allocation"
                    );
                } else {
                    tracing::trace!(
                        target: "kvbm-diag",
                        request_id = %request_id,
                        "get_num_new_matched_tokens → OnboardStaged (match already returned; defer poll)"
                    );
                }
                return Ok((None, false));
            };
            debug_assert!(
                (num_computed_tokens + num_external_tokens).is_multiple_of(self.block_size)
            );
            self.kvbm_metrics
                .matched_tokens
                .inc_by(num_external_tokens as u64);
            tracing::info!(
                target: "kvbm-diag",
                request_id = %request_id,
                num_external_tokens,
                num_computed_tokens,
                total_tokens = slot.sequence().total_tokens(),
                "get_num_new_matched_tokens → OnboardStaged (returning match)"
            );
            Ok((Some(num_external_tokens), true))
        } else {
            Ok((Some(0), false))
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
            let prefetched_host_ready = slot
                .as_any_mut()
                .downcast_mut::<VllmConnectorSlot>()
                .map(|s| -> Result<bool, SlotError> {
                    if s.try_stage_prefetched_host_matches()? {
                        return Ok(true);
                    }
                    Ok(s.has_staged_host_blocks())
                })
                .transpose()?
                .unwrap_or(false);
            if !prefetched_host_ready {
                anyhow::bail!(
                    "external tokens were reported before host-prefetch became CPU-ready for request {}",
                    request_id
                );
            }
            if let Some(slot) = slot.as_any_mut().downcast_mut::<VllmConnectorSlot>() {
                slot.clear_staged_match_report();
            }
            tracing::info!(
                target: "kvbm-diag",
                request_id = %request_id,
                num_external_tokens,
                "update_state_after_alloc → using prefetched host blocks"
            );
            let num_computed_tokens = block_ids.len() * self.block_size - num_external_tokens;
            tracing::info!(
                target: "kvbm-diag",
                request_id = %request_id,
                num_external_tokens,
                num_device_blocks = block_ids.len(),
                num_computed_tokens,
                block_size = self.block_size,
                "update_state_after_alloc → triggering onboarding"
            );
            slot.record_cached_device_tokens(num_computed_tokens);
            // NOTE: Do NOT advance_computed_position here.
            // vLLM's scheduler will report these tokens via num_computed_tokens
            // in apply_scheduler_output, which uses max(current_position, vllm_computed)
            // to advance. Advancing here would double-count: once from this call,
            // and again when vLLM's num_scheduled_tokens covers the remaining tokens
            // for the forward pass.
            slot.trigger_onboarding(num_external_tokens)?;
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
                    {
                        crate::lock_slot!(self, &key => slot);
                        if let Some(vllm_slot) =
                            slot.as_any_mut().downcast_mut::<VllmConnectorSlot>()
                        {
                            let _ = vllm_slot.release_prefetched_host_blocks();
                        }
                    }
                    let _ = self.slot_manager().remove_slot(&key);
                }
                self.active_slot_keys.remove(request_id);
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

        for request_id in &onboarding_slots {
            let _req_span = self.enter_request_span(request_id, "kvbm.flush_onboarding");
            let key = self.active_slot_key(request_id).ok_or_else(|| {
                anyhow::anyhow!(
                    "missing active SlotKey for onboarding request {}",
                    request_id
                )
            })?;
            crate::lock_slot!(self, &key => slot);
            crate::flush_slot_to_metadata!(slot, md, key);
            if !inflight_requests.remove(request_id) {
                tracing::warn!("request {request_id} not in inflight set (may have been cleared by clear_pool)");
            }
        }

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

            slot.apply_scheduler_output(
                &[],
                &[],
                new_req.num_computed_tokens,
                scheduled_tokens,
                None,
            )?;
            crate::flush_slot_to_metadata!(slot, md, key);
        }

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
            crate::lock_slot!(self, &key => slot);

            let scheduled_tokens = *scheduler_output
                .num_scheduled_tokens
                .get(request_id)
                .unwrap_or(&0);

            slot.apply_scheduler_output(
                &cached_req.new_token_ids,
                &cached_req.new_block_ids,
                cached_req.num_computed_tokens,
                scheduled_tokens,
                None,
            )?;

            if let Some(pending_ops) = slot.take_pending_operations() {
                md.add_operations_for_key(&key, pending_ops);
            }
        }

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

        serde_json::to_vec(&md)
            .map_err(|e| anyhow::anyhow!("Failed to serialize connector metadata: {}", e))
    }

    pub fn request_finished(
        &mut self,
        request_id: String,
        _block_ids: Vec<BlockId>,
    ) -> anyhow::Result<bool> {
        self.onboarding_slots.remove(&request_id);
        self.request_traces.remove(&request_id);

        let Some(key) = self.active_slot_key(&request_id) else {
            tracing::warn!(
                "request_finished called for request_id: {request_id} but no active slot key found"
            );
            self.inflight_requests.remove(&request_id);
            self.active_slot_keys.remove(&request_id);
            return Ok(false);
        };
        if !self.slot_manager().has_slot(&key) {
            tracing::warn!(
                "request_finished called for request_id: {request_id} but slot is not found"
            );
            self.inflight_requests.remove(&request_id);
            self.active_slot_keys.remove(&request_id);
            return Ok(false);
        }

        crate::lock_slot!(self, &key => slot);
        if matches!(slot.state(), SlotState::Onboarding(_))
            && let Some(vllm_slot) = slot.as_any_mut().downcast_mut::<VllmConnectorSlot>()
        {
            vllm_slot.discard_pending_operations();
        }

        slot.mark_as_finished(self.iteration_counter)?;
        self.inflight_requests.remove(&request_id);

        match slot.state() {
            SlotState::Finished => {
                if let Some(vllm_slot) = slot.as_any_mut().downcast_mut::<VllmConnectorSlot>() {
                    vllm_slot.release_prefetched_host_blocks()?;
                }
                self.active_slot_keys.remove(&request_id);
                self.slot_manager().remove_slot(&key)?;
            }
            SlotState::Finishing => {
                self.finishing_requests.insert(request_id);
            }
            _ => {
                if let Some(vllm_slot) = slot.as_any_mut().downcast_mut::<VllmConnectorSlot>() {
                    vllm_slot.release_prefetched_host_blocks()?;
                }
                self.active_slot_keys.remove(&request_id);
                self.slot_manager().remove_slot(&key)?;
            }
        }

        Ok(true)
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
        self.slot_manager().clear_pool(&pool)?;
        Ok(())
    }

    pub fn get_pool_status(&self) -> std::collections::HashMap<String, std::collections::HashMap<String, u64>> {
        self.slot_manager().get_pool_status()
    }
}
