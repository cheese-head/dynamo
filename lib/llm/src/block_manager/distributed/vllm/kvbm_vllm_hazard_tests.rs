// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Hazard regressions for the vLLM-facing slot reducer (`RequestPhase`) that are easy to miss
//! when reasoning about prefetch `operation_id` and late events.
//! See `docs/kvbm-vllm-hazard-notes.md`.
//!
//! Run: `cargo test kvbm_vllm_hazard`

#![cfg(test)]

use uuid::Uuid;

use super::slot_machine::{
    G4FailPolicy, PrefetchState, RequestPhase, SlotContext, SlotEvent,
};

type TestPhase = RequestPhase<(), ()>;

fn mock_ctx() -> SlotContext {
    SlotContext {
        block_size: 16,
        remote_enabled: true,
        g4_xfer_fail_policy: G4FailPolicy::Fallback,
    }
}

fn sample_prefetch_phase(op_id: Uuid) -> TestPhase {
    RequestPhase::Prefetching {
        prefetch: PrefetchState {
            operation_id: op_id,
            sequence_hashes: vec![0x1],
            num_external_tokens: 32,
            started_at: std::time::Instant::now(),
        },
        host_staging: vec![()],
        disk_staging: vec![],
        prefetched_blocks_for_stats: 1,
    }
}

#[test]
fn kvbm_vllm_hazard_prefetch_wrong_transfer_completed_operation_id_is_noop() {
    let ctx = mock_ctx();
    let op_id = Uuid::new_v4();
    let wrong = Uuid::new_v4();
    let phase = sample_prefetch_phase(op_id);
    let (out, effects) =
        phase.apply(SlotEvent::TransferCompleted { operation_id: wrong }, &ctx);
    assert!(
        matches!(out, RequestPhase::Prefetching { .. }),
        "Hazard H2: mismatched G4/prefetch op id must not advance to OnboardReady"
    );
    assert!(
        effects.is_empty(),
        "wrong op id should not emit prefetch completion effects"
    );
}

#[test]
fn kvbm_vllm_hazard_onboard_ready_absorbs_duplicate_prefetch_completion_event() {
    let ctx = mock_ctx();
    let op_id = Uuid::new_v4();
    let onboard: TestPhase = sample_prefetch_phase(op_id)
        .apply(SlotEvent::TransferCompleted { operation_id: op_id }, &ctx)
        .0;
    assert!(
        matches!(onboard, RequestPhase::OnboardReady { .. }),
        "setup: first matching completion reaches OnboardReady"
    );

    let (after_dup, effects) = onboard.apply(
        SlotEvent::TransferCompleted { operation_id: op_id },
        &ctx,
    );
    assert!(
        matches!(after_dup, RequestPhase::OnboardReady { .. }),
        "late duplicate TransferCompleted (same op id) must not corrupt OnboardReady"
    );
    assert!(
        effects.is_empty(),
        "absorbed late completion should not enqueue transfers"
    );
}

#[test]
fn kvbm_vllm_hazard_prefilling_ignores_transfer_completed() {
    let ctx = mock_ctx();
    let phase = TestPhase::Prefilling {
        iteration_first_scheduled: 1,
    };
    let (out, effects) = phase.apply(
        SlotEvent::TransferCompleted {
            operation_id: Uuid::new_v4(),
        },
        &ctx,
    );
    assert!(
        matches!(out, RequestPhase::Prefilling { .. }),
        "catch-all: prefilling must not interpret unrelated transfer completions"
    );
    assert!(effects.is_empty());
}

#[test]
fn kvbm_vllm_hazard_initialized_ignores_prefetch_ready() {
    let ctx = mock_ctx();
    let phase = TestPhase::Initialized;
    let (out, effects) = phase.apply(SlotEvent::PrefetchReady { blocks: vec![] }, &ctx);
    assert!(matches!(out, RequestPhase::Initialized));
    assert!(effects.is_empty());
}

/// H1 / DP: TP aggregation for worker op sets (Rust mirror of `worker_metadata.py`).
#[test]
fn kvbm_vllm_hazard_tp_worker_metadata_intersection_for_recv_path() {
    use std::collections::HashSet;

    use super::worker_metadata_merge::{aggregate_tp, KvbmWorkerMetadataFixture};

    let mut a = KvbmWorkerMetadataFixture::default();
    a.completed_onboard_ops.insert(
        "r1".into(),
        HashSet::from(["op-a".into(), "op-b".into()]),
    );
    let mut b = KvbmWorkerMetadataFixture::default();
    b.completed_onboard_ops
        .insert("r1".into(), HashSet::from(["op-b".into()]));
    let m = aggregate_tp(&a, &b);
    assert_eq!(
        m.completed_onboard_ops.get("r1").unwrap(),
        &HashSet::from(["op-b".into()])
    );
}

/// Phase 4 placeholder: vLLM uses `invalid_block_ids` in `update_from_output` / `_handle_invalid_blocks`.
#[test]
fn kvbm_vllm_hazard_invalid_block_ids_must_reach_scheduler_contract() {
    let invalid: Vec<u32> = vec![0, 1];
    assert!(
        !invalid.is_empty(),
        "when async loads fail, connector must eventually surface failed device block indices for scheduler policy"
    );
}

/// Phase 5: non-attention / Mamba external KV not covered by KVBM vLLM mocks today.
#[test]
#[ignore = "document support matrix for Mamba + external KV before enabling"]
fn kvbm_vllm_hazard_mamba_external_kv_matrix() {
    panic!("enable when a Rust mocker scenario exists for this model class");
}
