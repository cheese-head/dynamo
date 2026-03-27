// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! **Full vLLM connector E2E tests — no vLLM process required.**
//!
//! Exercises the production call sequence that vLLM drives via Python bindings:
//!
//!   `create_slot` → `get_num_new_matched_tokens` → `update_state_after_alloc`
//!   → `build_connector_metadata` (with `SchedulerOutput`) → `apply_worker_feedback`
//!   → `request_finished`
//!
//! Uses real `KvbmLeader` + `KvbmWorker` (ZMQ), real `KvBlockManager` (CUDA device,
//! host, disk), real `ConnectorSlotManager` (spawns `LocalTransferEngine`), and real
//! `KvConnectorLeaderCore`.  The only thing missing is the Python runtime.
//!
//! **Threading model:** In production, Python calls leader-core methods from a
//! non-Tokio thread.  The effect executor uses `_blocking` pool lookups that
//! internally `block_on()`.  We replicate this by running all synchronous
//! leader-core calls inside `spawn_blocking` to avoid "cannot block in async"
//! panics from `managed.rs`.
//!
//! Feature gate: `testing-cuda` (requires GPU).
//!
//! Run:
//! ```sh
//! ./kvbm-tests.sh -- 'cargo test -p dynamo-llm vllm_connector_e2e --features testing-cuda -- --nocapture'
//! ```

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use rstest::rstest;
use serial_test::serial;

use crate::block_manager::{
    distributed::{
        KvbmWorker,
        tests::{build_leader_and_workers, build_test_block_manager},
    },
    metrics_kvbm::{KvbmMetrics, KvbmMetricsRegistry},
};

use super::{
    CachedRequestData, ConnectorMetadata, ConnectorSlotManager,
    KvConnectorLeaderCore, NewRequestData, SchedulerOutput,
};

use dynamo_runtime::logging::init as init_logging;

const BLOCK_SIZE: usize = 4;
const NUM_BLOCKS: usize = 8;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Wraps the leader core + workers.  All synchronous leader-core operations
/// must run via [`on_blocking`] to avoid Tokio `block_on`-inside-runtime panics.
struct VllmE2eStack {
    inner: Arc<Mutex<VllmE2eInner>>,
}

struct VllmE2eInner {
    core: KvConnectorLeaderCore,
    _workers: Vec<KvbmWorker>,
}

impl VllmE2eStack {
    async fn new() -> anyhow::Result<Self> {
        let (leader, workers) =
            build_leader_and_workers(1, NUM_BLOCKS, NUM_BLOCKS, NUM_BLOCKS, BLOCK_SIZE).await?;
        let leader = Arc::new(leader);
        let block_manager =
            build_test_block_manager(leader.clone(), NUM_BLOCKS, BLOCK_SIZE).await?;

        let kvbm_metrics = KvbmMetrics::new(&KvbmMetricsRegistry::default(), false, 0);
        let slot_manager_cell = Arc::new(OnceLock::new());
        let slot_manager =
            ConnectorSlotManager::new(block_manager, leader.clone(), kvbm_metrics.clone(), None);
        slot_manager_cell
            .set(slot_manager)
            .map_err(|_| anyhow::anyhow!("OnceLock already set"))?;

        let core =
            KvConnectorLeaderCore::new(slot_manager_cell, leader, BLOCK_SIZE, kvbm_metrics);

        Ok(Self {
            inner: Arc::new(Mutex::new(VllmE2eInner {
                core,
                _workers: workers,
            })),
        })
    }

    /// Run a closure that accesses the leader core on a blocking OS thread,
    /// replicating how Python calls the connector from outside Tokio.
    async fn on_blocking<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&mut KvConnectorLeaderCore) -> R + Send + 'static,
        R: Send + 'static,
    {
        let inner = self.inner.clone();
        tokio::task::spawn_blocking(move || {
            let mut guard = inner.lock().unwrap();
            f(&mut guard.core)
        })
        .await
        .unwrap()
    }

    /// Drop the inner state on a blocking thread while the Tokio runtime is
    /// still alive.  This triggers `ConnectorSlotManager::drop` → `cancel` +
    /// `detach` for the `LocalTransferEngine` task, preventing the
    /// "Critical task was not detached prior to drop!" panic during runtime
    /// shutdown.
    async fn shutdown(self) {
        let inner = self.inner;
        tokio::task::spawn_blocking(move || {
            drop(inner);
        })
        .await
        .unwrap();
    }
}

fn make_tokens(n: usize) -> Vec<u32> {
    (1..=n as u32).collect()
}

fn make_scheduler_output(
    new_reqs: Vec<NewRequestData>,
    cached_reqs: Vec<CachedRequestData>,
) -> SchedulerOutput {
    let mut num_scheduled = HashMap::new();
    for r in &new_reqs {
        num_scheduled.insert(r.request_id.clone(), BLOCK_SIZE);
    }
    for r in &cached_reqs {
        num_scheduled.insert(r.request_id.clone(), BLOCK_SIZE);
    }
    SchedulerOutput {
        new_requests: new_reqs,
        cached_requests: cached_reqs,
        num_scheduled_tokens: num_scheduled,
    }
}

fn build_and_parse_metadata(
    core: &mut KvConnectorLeaderCore,
    so: SchedulerOutput,
) -> anyhow::Result<ConnectorMetadata> {
    let bytes = core.build_connector_metadata(so)?;
    let md: ConnectorMetadata = serde_json::from_slice(&bytes)?;
    Ok(md)
}

// ---------------------------------------------------------------------------
// Scenario 1: Simple prefill → decode → finish (no cache hit)
// ---------------------------------------------------------------------------

#[rstest]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[serial]
async fn e2e_vllm_connector_prefill_decode_finish() -> anyhow::Result<()> {
    init_logging();
    let stack = VllmE2eStack::new().await?;

    let req_id = "req-prefill-decode-1";
    let num_prompt_tokens = BLOCK_SIZE * 2;

    stack
        .on_blocking(move |core| -> anyhow::Result<()> {
            // 1. create_slot
            core.create_slot(req_id.to_string(), 0, make_tokens(num_prompt_tokens))?;

            // 2. get_num_new_matched_tokens — first poll triggers AcquireMatches.
            //    Remote-candidates fallback may produce OnboardReady even with
            //    empty host/disk pools, so handle both outcomes like vLLM would.
            let (matched, needs_alloc) =
                core.get_num_new_matched_tokens(req_id.to_string(), num_prompt_tokens, 0)?;
            assert!(matched.is_some(), "first poll should be definitive");

            if needs_alloc {
                let num_external = matched.unwrap();
                let num_alloc_blocks = (num_external + BLOCK_SIZE - 1) / BLOCK_SIZE;
                let alloc_ids: Vec<usize> = (0..num_alloc_blocks).collect();
                core.update_state_after_alloc(req_id.to_string(), alloc_ids, num_external)?;
            }

            // 3. build_connector_metadata with new-request entry (prefill)
            let md = build_and_parse_metadata(
                core,
                make_scheduler_output(
                    vec![NewRequestData {
                        request_id: req_id.to_string(),
                        prompt_token_ids: make_tokens(num_prompt_tokens),
                        block_ids: vec![0, 1],
                        num_computed_tokens: 0,
                        priorities: None,
                    }],
                    vec![],
                ),
            )?;
            assert_eq!(md.iteration, 1);
            assert!(!md.new_slots.is_empty(), "should create slot in metadata");
            assert_eq!(md.new_slots[0].key.request_id, req_id);

            // 4. Decode step: cached_request with new token
            let md2 = build_and_parse_metadata(
                core,
                make_scheduler_output(
                    vec![],
                    vec![CachedRequestData {
                        request_id: req_id.to_string(),
                        resumed_from_preemption: false,
                        new_token_ids: vec![99],
                        new_block_ids: vec![],
                        num_computed_tokens: BLOCK_SIZE,
                        priorities: None,
                    }],
                ),
            )?;
            assert_eq!(md2.iteration, 2);

            // 5. request_finished
            let cleaned = core.request_finished(req_id.to_string(), vec![0, 1])?;
            assert!(cleaned, "slot should be cleaned up");
            assert!(!core.has_slot(req_id), "slot removed");

            Ok(())
        })
        .await?;

    stack.shutdown().await;
    Ok(())
}

// ---------------------------------------------------------------------------
// Scenario 2: Cache hit → onboard → decode → offload effects → finish
// ---------------------------------------------------------------------------

#[rstest]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[serial]
async fn e2e_vllm_connector_cache_hit_onboard_decode_offload() -> anyhow::Result<()> {
    init_logging();
    let stack = VllmE2eStack::new().await?;

    let prompt_len = BLOCK_SIZE * 2;

    // --- Phase A: donor request exercises full poll → schedule → finish path ---
    // The first poll may return a cache hit (remote_candidates trigger the
    // remote-lookup fallback path → OnboardReady) depending on block manager
    // configuration.  Handle both outcomes the way vLLM would.
    stack
        .on_blocking(move |core| -> anyhow::Result<()> {
            let donor = "req-donor";
            core.create_slot(donor.to_string(), 0, make_tokens(prompt_len))?;

            let (matched, needs_alloc) =
                core.get_num_new_matched_tokens(donor.to_string(), prompt_len, 0)?;
            assert!(matched.is_some(), "donor poll should be definitive");

            if needs_alloc {
                let num_external = matched.unwrap();
                let num_alloc_blocks = (num_external + BLOCK_SIZE - 1) / BLOCK_SIZE;
                let alloc_ids: Vec<usize> = (0..num_alloc_blocks).collect();
                core.update_state_after_alloc(donor.to_string(), alloc_ids, num_external)?;
            }

            build_and_parse_metadata(
                core,
                make_scheduler_output(
                    vec![NewRequestData {
                        request_id: donor.to_string(),
                        prompt_token_ids: make_tokens(prompt_len),
                        block_ids: vec![0, 1],
                        num_computed_tokens: 0,
                        priorities: None,
                    }],
                    vec![],
                ),
            )?;

            Ok(())
        })
        .await?;

    // Give the transfer engine time to offload donor blocks.
    tokio::time::sleep(std::time::Duration::from_millis(300)).await;

    stack
        .on_blocking(move |core| -> anyhow::Result<()> {
            core.request_finished("req-donor".to_string(), vec![0, 1])?;
            Ok(())
        })
        .await?;

    tokio::time::sleep(std::time::Duration::from_millis(200)).await;

    // --- Phase B: second request with same prefix ---
    stack
        .on_blocking(move |core| -> anyhow::Result<()> {
            let req_id = "req-cache-hit-1";
            core.create_slot(req_id.to_string(), 0, make_tokens(prompt_len))?;

            let (matched, needs_alloc) =
                core.get_num_new_matched_tokens(req_id.to_string(), prompt_len, 0)?;
            assert!(matched.is_some(), "poll should return Some, got None");

            if needs_alloc {
                let num_external = matched.unwrap();
                let num_alloc_blocks = (num_external + BLOCK_SIZE - 1) / BLOCK_SIZE;
                let alloc_ids: Vec<usize> = (2..2 + num_alloc_blocks).collect();
                core.update_state_after_alloc(req_id.to_string(), alloc_ids, num_external)?;
            }

            let new_block_ids = if needs_alloc { vec![2, 3] } else { vec![0, 1] };
            build_and_parse_metadata(
                core,
                make_scheduler_output(
                    vec![NewRequestData {
                        request_id: req_id.to_string(),
                        prompt_token_ids: make_tokens(prompt_len),
                        block_ids: new_block_ids.clone(),
                        num_computed_tokens: 0,
                        priorities: None,
                    }],
                    vec![],
                ),
            )?;

            // Decode step with priorities → EnqueueOffloadTransfer
            let md_decode = build_and_parse_metadata(
                core,
                make_scheduler_output(
                    vec![],
                    vec![CachedRequestData {
                        request_id: req_id.to_string(),
                        resumed_from_preemption: false,
                        new_token_ids: vec![200, 201, 202, 203],
                        new_block_ids: vec![4],
                        num_computed_tokens: BLOCK_SIZE,
                        priorities: Some(vec![1, 2, 3]),
                    }],
                ),
            )?;
            tracing::info!(
                iteration = md_decode.iteration,
                num_ops = md_decode.operations.len(),
                "decode metadata produced"
            );

            let mut finish_ids = new_block_ids;
            finish_ids.push(4);
            core.request_finished(req_id.to_string(), finish_ids)?;
            Ok(())
        })
        .await?;

    stack.shutdown().await;
    Ok(())
}

// ---------------------------------------------------------------------------
// Scenario 3: Worker feedback (completed + failed) + hazard counters
// ---------------------------------------------------------------------------

#[rstest]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[serial]
async fn e2e_vllm_connector_worker_feedback_and_hazards() -> anyhow::Result<()> {
    init_logging();
    let stack = VllmE2eStack::new().await?;

    stack
        .on_blocking(move |core| -> anyhow::Result<()> {
            let req_id = "req-feedback-1";
            core.create_slot(req_id.to_string(), 0, make_tokens(BLOCK_SIZE))?;

            let _ = core.get_num_new_matched_tokens(req_id.to_string(), BLOCK_SIZE, 0)?;

            build_and_parse_metadata(
                core,
                make_scheduler_output(
                    vec![NewRequestData {
                        request_id: req_id.to_string(),
                        prompt_token_ids: make_tokens(BLOCK_SIZE),
                        block_ids: vec![0],
                        num_computed_tokens: 0,
                        priorities: None,
                    }],
                    vec![],
                ),
            )?;

            // Completed op for a real slot (uuid won't match, but shouldn't panic)
            let fake_op_id = uuid::Uuid::new_v4();
            let mut completed = HashMap::new();
            completed.insert(req_id.to_string(), vec![fake_op_id]);
            core.apply_worker_feedback(&completed, &HashMap::new());

            // Failed op for unknown request → hazard H5
            let mut failed = HashMap::new();
            failed.insert("nonexistent-req".to_string(), vec![uuid::Uuid::new_v4()]);
            core.apply_worker_feedback(&HashMap::new(), &failed);

            let stats = core.worker_feedback_stats_snapshot();
            assert_eq!(stats.unknown_request_id, 1, "H5: one unknown request");

            core.request_finished(req_id.to_string(), vec![0])?;
            Ok(())
        })
        .await?;

    stack.shutdown().await;
    Ok(())
}

// ---------------------------------------------------------------------------
// Scenario 4: Preemption round-trip
// ---------------------------------------------------------------------------

#[rstest]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[serial]
async fn e2e_vllm_connector_preemption_and_resume() -> anyhow::Result<()> {
    init_logging();
    let stack = VllmE2eStack::new().await?;

    let prompt_len = BLOCK_SIZE * 2;

    stack
        .on_blocking(move |core| -> anyhow::Result<()> {
            let req_id = "req-preempt-1";
            core.create_slot(req_id.to_string(), 0, make_tokens(prompt_len))?;

            let _ = core.get_num_new_matched_tokens(req_id.to_string(), prompt_len, 0)?;

            build_and_parse_metadata(
                core,
                make_scheduler_output(
                    vec![NewRequestData {
                        request_id: req_id.to_string(),
                        prompt_token_ids: make_tokens(prompt_len),
                        block_ids: vec![0, 1],
                        num_computed_tokens: 0,
                        priorities: None,
                    }],
                    vec![],
                ),
            )?;

            // Preempt
            core.handle_preemptions_via_machine(&[req_id.to_string()]);

            // Resume via cached_request with resumed_from_preemption=true
            let md_resume = build_and_parse_metadata(
                core,
                make_scheduler_output(
                    vec![],
                    vec![CachedRequestData {
                        request_id: req_id.to_string(),
                        resumed_from_preemption: true,
                        new_token_ids: vec![],
                        new_block_ids: vec![],
                        num_computed_tokens: 0,
                        priorities: None,
                    }],
                ),
            )?;
            tracing::info!(iteration = md_resume.iteration, "preemption resume");

            // Re-poll after resume → AcquireMatches restarts
            let (matched, _) =
                core.get_num_new_matched_tokens(req_id.to_string(), prompt_len, 0)?;
            assert!(matched.is_some(), "post-preemption poll should be definitive");

            core.request_finished(req_id.to_string(), vec![0, 1])?;
            Ok(())
        })
        .await?;

    stack.shutdown().await;
    Ok(())
}

// ---------------------------------------------------------------------------
// Scenario 5: Multiple concurrent requests
// ---------------------------------------------------------------------------

#[rstest]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[serial]
async fn e2e_vllm_connector_multi_request_scheduling() -> anyhow::Result<()> {
    init_logging();
    let stack = VllmE2eStack::new().await?;

    let prompt_len = BLOCK_SIZE;

    stack
        .on_blocking(move |core| -> anyhow::Result<()> {
            let req_a = "req-multi-a";
            let req_b = "req-multi-b";

            core.create_slot(req_a.to_string(), 0, make_tokens(prompt_len))?;
            core.create_slot(req_b.to_string(), 0, make_tokens(prompt_len))?;

            let _ = core.get_num_new_matched_tokens(req_a.to_string(), prompt_len, 0)?;
            let _ = core.get_num_new_matched_tokens(req_b.to_string(), prompt_len, 0)?;

            // Schedule both as new in the same iteration
            let md = build_and_parse_metadata(
                core,
                make_scheduler_output(
                    vec![
                        NewRequestData {
                            request_id: req_a.to_string(),
                            prompt_token_ids: make_tokens(prompt_len),
                            block_ids: vec![0],
                            num_computed_tokens: 0,
                            priorities: None,
                        },
                        NewRequestData {
                            request_id: req_b.to_string(),
                            prompt_token_ids: make_tokens(prompt_len),
                            block_ids: vec![1],
                            num_computed_tokens: 0,
                            priorities: None,
                        },
                    ],
                    vec![],
                ),
            )?;
            assert_eq!(md.new_slots.len(), 2, "both requests in metadata");

            // Finish one, keep the other
            core.request_finished(req_a.to_string(), vec![0])?;
            assert!(!core.has_slot(req_a));
            assert!(core.has_slot(req_b));

            // Decode step for req_b
            let md2 = build_and_parse_metadata(
                core,
                make_scheduler_output(
                    vec![],
                    vec![CachedRequestData {
                        request_id: req_b.to_string(),
                        resumed_from_preemption: false,
                        new_token_ids: vec![42],
                        new_block_ids: vec![],
                        num_computed_tokens: BLOCK_SIZE,
                        priorities: None,
                    }],
                ),
            )?;
            assert_eq!(md2.iteration, 2);

            core.request_finished(req_b.to_string(), vec![1])?;
            assert!(!core.has_slot(req_b));

            Ok(())
        })
        .await?;

    stack.shutdown().await;
    Ok(())
}
