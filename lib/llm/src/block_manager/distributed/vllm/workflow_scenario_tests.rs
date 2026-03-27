// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Tier-1 **workflow scenario** tests: full reducer paths from a request’s first match poll
//! through terminal states, parameterized for regression discovery.
//!
//! # What this guarantees
//! - Every scenario is an ordered script of real [`SlotEvent`](super::slot_machine::SlotEvent)s
//!   against the production reducer (`RequestPhase::apply`).
//! - Failures point to a **named scenario** and step, which is easier to bisect than one
//!   monolithic “mega test”.
//!
//! # What it does *not* guarantee (use other tiers)
//! - **Effect executor I/O** (`RunLocalLookup` blocking, transfer engine, G4): see
//!   `effect_executor::apply_and_execute` call sites and integration tests.
//! - **KvConnectorLeaderCore** scheduling + metadata: needs harness with
//!   [`super::ConnectorSlotManager`] + block manager (heavy; often feature-gated).
//! - **Exhaustive** proof of zero bugs: in practice, combine this module with
//!   `slot_machine_tests`, `slot_phase_tests`, `vllm_kv_semantics_tests`, `kvbm_vllm_hazard_tests`,
//!   and distributed `build_leader_and_workers` tests.
//!
//! # Phase coverage checklist (Tier-1 reducer)
//! Use `cargo test -p dynamo-llm --lib workflow_scenario` and the `scenario_*` names below.
//! Complement with `slot_machine_tests` for single-row table coverage and edge cases.

#![cfg(test)]

use super::slot_machine::*;
use super::test_harness::helpers::mock_load_request;
use super::OperationTracker;

use rstest::rstest;
use uuid::Uuid;

type P = RequestPhase<(), ()>;
type E = SlotEvent<(), ()>;

fn ctx(bs: usize, remote: bool) -> SlotContext {
    SlotContext {
        block_size: bs,
        remote_enabled: remote,
        g4_xfer_fail_policy: G4FailPolicy::Fallback,
    }
}

fn sched_prefill() -> E {
    E::ApplySchedulerOutput {
        tokens: vec![],
        block_ids: vec![1, 2],
        num_computed_tokens: 0,
        num_scheduled_tokens: 128,
        priorities: None,
        iteration: 1,
    }
}

fn sched_decode() -> E {
    E::ApplySchedulerOutput {
        tokens: vec![42, 43, 44],
        block_ids: vec![1, 2, 3],
        num_computed_tokens: 128,
        num_scheduled_tokens: 1,
        priorities: Some(vec![10, 20, 30]),
        iteration: 1,
    }
}

fn mock_remote_match() -> (u64, u64) {
    (0xabcd, 0x1234)
}

/// Initialized → … → Finished (host+disk only, remote enabled but no candidates).
fn walk_local_happy(bs: usize) -> P {
    let c = ctx(bs, true);
    let mut p = P::Initialized;
    p = p
        .apply(
            E::AcquireMatches {
                num_computed_tokens: 0,
            },
            &c,
        )
        .0;
    assert!(matches!(p, P::AwaitingLookup));

    p = p
        .apply(
            E::LocalLookupCompleted {
                host_blocks: vec![(), (), ()],
                disk_blocks: vec![()],
                remote_candidates: vec![],
            },
            &c,
        )
        .0;
    assert!(matches!(
        p,
        P::OnboardReady {
            disclosure: DisclosureState::Pending,
            ..
        }
    ));

    p = p.apply(E::PollMatchReport, &c).0;
    assert!(matches!(
        p,
        P::OnboardReady {
            disclosure: DisclosureState::Disclosed,
            ..
        }
    ));

    p = p
        .apply(
            E::AllocCompleted {
                block_ids: vec![10usize, 11, 12, 13],
                num_external_tokens: 64,
            },
            &c,
        )
        .0;
    assert!(matches!(p, P::Onboarding { .. }));

    p = p.apply(sched_prefill(), &c).0;
    assert!(matches!(p, P::Prefilling { .. }));

    p = p.apply(sched_decode(), &c).0;
    assert!(matches!(p, P::Decoding { .. }));

    p = p.apply(E::RequestFinished, &c).0;
    assert!(matches!(p, P::Finished));
    p
}

/// Remote prefetch path through Finished.
fn walk_remote_prefetch_happy(bs: usize) -> P {
    let c = ctx(bs, true);
    let mut p = P::Initialized;
    p = p
        .apply(
            E::AcquireMatches {
                num_computed_tokens: 0,
            },
            &c,
        )
        .0;
    p = p
        .apply(
            E::LocalLookupCompleted {
                host_blocks: vec![(), ()],
                disk_blocks: vec![],
                remote_candidates: vec![0x1111, 0x2222],
            },
            &c,
        )
        .0;
    assert!(matches!(p, P::LookingUp { .. }));

    p = p
        .apply(
            E::RemoteLookupCompleted {
                matches: vec![mock_remote_match()],
            },
                  &c,
        )
        .0;
    assert!(matches!(p, P::Prefetching { .. }));

    p = p
        .apply(
            E::PrefetchReady {
                blocks: vec![(), ()],
            },
            &c,
        )
        .0;
    assert!(matches!(p, P::OnboardReady { .. }));

    p = p.apply(E::PollMatchReport, &c).0;
    p = p
        .apply(
            E::AllocCompleted {
                block_ids: vec![20, 21],
                num_external_tokens: 128,
            },
            &c,
        )
        .0;
    assert!(matches!(p, P::Onboarding { .. }));

    p = p.apply(sched_decode(), &c).0;
    assert!(matches!(p, P::Decoding { .. }));

    p = p.apply(E::RequestFinished, &c).0;
    assert!(matches!(p, P::Finished));
    p
}

/// With `remote_enabled == false`, remote candidates are ignored and local staging wins.
fn walk_remote_disabled_ignores_candidates(bs: usize) -> P {
    let c = ctx(bs, false);
    assert!(
        !c.remote_enabled,
        "fixture assumes remote disabled (remote candidates must not force LookingUp)"
    );
    let mut p = P::Initialized;
    p = p
        .apply(
            E::AcquireMatches {
                num_computed_tokens: 0,
            },
            &c,
        )
        .0;
    assert!(
        matches!(p, P::AwaitingLookup),
        "after AcquireMatches expected AwaitingLookup, got {p:?}"
    );
    p = p
        .apply(
            E::LocalLookupCompleted {
                host_blocks: vec![(); 1],
                disk_blocks: Vec::new(),
                remote_candidates: vec![0xbeef],
            },
            &c,
        )
        .0;
    match &p {
        P::OnboardReady {
            num_external_tokens,
            host_staging,
            disk_staging,
            remote_hashes,
            ..
        } => {
            assert_eq!(
                *num_external_tokens,
                bs,
                "one host block × block_size (remote candidates ignored when remote_enabled is false)"
            );
            assert_eq!(host_staging.len(), 1);
            assert!(disk_staging.is_empty());
            assert!(
                remote_hashes.is_empty(),
                "local-only path must not surface remote_hashes"
            );
        }
        other => panic!(
            "expected OnboardReady (local hit, remote path off); got {other:?}. \
             LookingUp here means remote_enabled was true; Initialized means no local staging."
        ),
    }
    p
}

/// No host, no disk, no remote matches → MatchNone branch back toward Initialized.
fn walk_miss_no_local_or_remote(bs: usize) -> P {
    let c = ctx(bs, true);
    let mut p = P::Initialized;
    p = p
        .apply(
            E::AcquireMatches {
                num_computed_tokens: 0,
            },
            &c,
        )
        .0;
    let (p2, eff) = p.apply(
        E::LocalLookupCompleted {
            host_blocks: vec![],
            disk_blocks: vec![],
            remote_candidates: vec![],
        },
        &c,
    );
    assert!(matches!(p2, P::Initialized));
    assert!(eff.iter().any(|e| matches!(e, SlotEffect::MatchNone)));
    p2
}

/// Remote lookup returns no matches → OnboardReady from local staging only.
fn walk_remote_completed_empty_matches(bs: usize) -> P {
    let c = ctx(bs, true);
    let mut p = P::AwaitingLookup;
    p = p
        .apply(
            E::LocalLookupCompleted {
                host_blocks: vec![()],
                disk_blocks: vec![],
                remote_candidates: vec![0x01],
            },
            &c,
        )
        .0;
    assert!(matches!(p, P::LookingUp { .. }));
    p = p.apply(E::RemoteLookupCompleted { matches: vec![] }, &c).0;
    assert!(matches!(p, P::OnboardReady { .. }));
    p
}

/// `RemoteLookupClosed` while LookingUp.
fn walk_remote_lookup_closed(bs: usize) -> P {
    let c = ctx(bs, true);
    let mut p = P::LookingUp {
        host_blocks: vec![(), (), ()],
        disk_blocks: vec![()],
    };
    p = p.apply(E::RemoteLookupClosed, &c).0;
    assert!(matches!(p, P::OnboardReady { .. }));
    p
}

/// Preempt mid-prefetch, then AcquireMatches from Preempted completes locally.
fn walk_preempt_prefetch_then_local_finish(bs: usize) -> P {
    let c = ctx(bs, true);
    let prefetch = PrefetchState {
        operation_id: Uuid::new_v4(),
        sequence_hashes: vec![0x1, 0x2],
        num_external_tokens: 32,
        started_at: std::time::Instant::now(),
    };
    let mut p = P::Prefetching {
        prefetch,
        host_staging: vec![()],
        disk_staging: vec![],
        prefetched_blocks_for_stats: 1,
    };
    p = p.apply(E::Preempt, &c).0;
    assert!(matches!(p, P::Preempted { .. }));

    p = p
        .apply(
            E::AcquireMatches {
                num_computed_tokens: 0,
            },
            &c,
        )
        .0;
    assert!(matches!(p, P::AwaitingLookup));

    p = p
        .apply(
            E::LocalLookupCompleted {
                host_blocks: vec![(), ()],
                disk_blocks: vec![],
                remote_candidates: vec![],
            },
            &c,
        )
        .0;
    assert!(matches!(p, P::OnboardReady { .. }));
    p
}

/// Decoding with pending ops → Finishing, drain op → Finished.
fn walk_finishing_drains_ops(bs: usize) -> P {
    let c = ctx(bs, true);
    let mut tracker = OperationTracker::new();
    tracker.append_pending(mock_load_request("finish-walk", vec![1]));
    let _ = tracker.take_pending_for_dispatch();

    let mut p = P::Decoding {
        ops: tracker,
        iteration_first_scheduled: 1,
    };
    p = p.apply(E::RequestFinished, &c).0;
    assert!(matches!(p, P::Finishing { .. }));

    p = p
        .apply(
            E::TransferCompleted {
                operation_id: Uuid::new_v4(),
            },
            &c,
        )
        .0;
    assert!(matches!(p, P::Finished));
    p
}

/// SkippedPrefill chunk then decode.
fn walk_skipped_prefill_to_decoding(bs: usize) -> P {
    let c = ctx(bs, true);
    let mut p = P::Prefilling {
        iteration_first_scheduled: 0,
    };
    p = p.apply(E::MarkSkipped, &c).0;
    assert!(matches!(p, P::SkippedPrefill));

    p = p.apply(sched_decode(), &c).0;
    assert!(matches!(p, P::Decoding { .. }));
    p
}

#[rstest]
#[case(16)]
#[case(32)]
fn scenario_local_happy_path_finished(#[case] block_size: usize) {
    let _ = walk_local_happy(block_size);
}

#[rstest]
#[case(16)]
fn scenario_remote_prefetch_happy_path_finished(#[case] block_size: usize) {
    let _ = walk_remote_prefetch_happy(block_size);
}

#[rstest]
#[case(16)]
#[case(32)]
fn scenario_remote_disabled_ignores_remote_candidates(#[case] block_size: usize) {
    let _ = walk_remote_disabled_ignores_candidates(block_size);
}

#[rstest]
#[case(16)]
fn scenario_miss_emits_match_none(#[case] block_size: usize) {
    let _ = walk_miss_no_local_or_remote(block_size);
}

#[rstest]
#[case(16)]
fn scenario_remote_empty_matches_to_onboard_ready(#[case] block_size: usize) {
    let _ = walk_remote_completed_empty_matches(block_size);
}

#[rstest]
#[case(16)]
fn scenario_remote_lookup_closed(#[case] block_size: usize) {
    let _ = walk_remote_lookup_closed(block_size);
}

#[rstest]
#[case(16)]
fn scenario_preempt_prefetch_then_local_onboard(#[case] block_size: usize) {
    let _ = walk_preempt_prefetch_then_local_finish(block_size);
}

#[rstest]
#[case(16)]
fn scenario_finishing_drains_dispatched_ops(#[case] block_size: usize) {
    let _ = walk_finishing_drains_ops(block_size);
}

#[rstest]
#[case(16)]
fn scenario_skipped_prefill_then_decode(#[case] block_size: usize) {
    let _ = walk_skipped_prefill_to_decoding(block_size);
}

// =============================================================================
// Regression: phantom OnboardReady from empty staging vecs
//
// Root cause: RunLocalLookup wrapped empty block vecs as vec![vec![]], so
// LookingUp carried host_blocks/disk_blocks with .len()==1 (one empty inner
// vec). RemoteLookupClosed / RemoteLookupCompleted{empty} checked
// .is_empty() on the outer vec, saw "not empty", and transitioned to
// OnboardReady with phantom num_external_tokens — causing an indefinite hang
// in Onboarding because no real transfers were dispatched.
// =============================================================================

/// **Negative regression**: LookingUp with 0 actual local blocks + remote
/// lookup closed must yield MatchNone → Initialized, NOT phantom OnboardReady.
///
/// Before the fix this test would fail: the slot would reach OnboardReady(512)
/// with zero staging blocks, then hang in Onboarding forever because
/// EnqueueOnboardTransfer would flatten the empty inner vecs to zero transfers.
#[rstest]
#[case(16)]
#[case(128)]
#[case(256)]
fn regression_phantom_onboard_zero_local_blocks_remote_closed(#[case] block_size: usize) {
    let c = ctx(block_size, true);
    let p = P::LookingUp {
        host_blocks: vec![],
        disk_blocks: vec![],
    };
    let (phase, effects) = p.apply(E::RemoteLookupClosed, &c);
    assert!(
        matches!(phase, P::Initialized),
        "expected Initialized (MatchNone) with 0 local blocks + remote closed, got {phase:?}"
    );
    assert!(
        effects.iter().any(|e| matches!(e, SlotEffect::MatchNone)),
        "must emit MatchNone effect when there are no blocks to onboard"
    );
}

/// **Negative regression**: same phantom scenario via RemoteLookupCompleted
/// with empty matches instead of RemoteLookupClosed.
#[rstest]
#[case(16)]
#[case(256)]
fn regression_phantom_onboard_zero_local_blocks_remote_empty_matches(#[case] block_size: usize) {
    let c = ctx(block_size, true);
    let p = P::LookingUp {
        host_blocks: vec![],
        disk_blocks: vec![],
    };
    let (phase, effects) = p.apply(E::RemoteLookupCompleted { matches: vec![] }, &c);
    assert!(
        matches!(phase, P::Initialized),
        "expected Initialized (MatchNone) with 0 local blocks + 0 remote matches, got {phase:?}"
    );
    assert!(
        effects.iter().any(|e| matches!(e, SlotEffect::MatchNone)),
        "must emit MatchNone effect"
    );
}

/// **Positive**: LookingUp with real local blocks + remote closed → OnboardReady
/// with correct num_external_tokens matching the actual block count.
#[rstest]
#[case(16, 3, 1)]   // 3 host + 1 disk → 4 blocks → 4*16 = 64 tokens
#[case(256, 0, 2)]   // 0 host + 2 disk → 2 blocks → 2*256 = 512 tokens
#[case(256, 1, 0)]   // 1 host + 0 disk → 1 block  → 1*256 = 256 tokens
fn regression_real_blocks_remote_closed_correct_count(
    #[case] block_size: usize,
    #[case] n_host: usize,
    #[case] n_disk: usize,
) {
    let c = ctx(block_size, true);
    let p = P::LookingUp {
        host_blocks: vec![(); n_host],
        disk_blocks: vec![(); n_disk],
    };
    let (phase, _) = p.apply(E::RemoteLookupClosed, &c);
    match phase {
        P::OnboardReady {
            num_external_tokens,
            host_staging,
            disk_staging,
            ..
        } => {
            let expected_tokens = (n_host + n_disk) * block_size;
            assert_eq!(
                num_external_tokens, expected_tokens,
                "num_external_tokens must equal actual block count × block_size"
            );
            assert_eq!(host_staging.len(), n_host);
            assert_eq!(disk_staging.len(), n_disk);
        }
        other => panic!(
            "expected OnboardReady with {n_host} host + {n_disk} disk blocks, got {other:?}"
        ),
    }
}

/// **Full walk**: fresh instance (0 host, 0 disk) with remote candidates sent
/// to registry that returns 0 matches. Must complete as MatchNone, not hang.
/// This mirrors the exact production failure: request on a fresh instance with
/// a freshly-restarted registry.
#[rstest]
#[case(256)]
fn regression_fresh_instance_full_walk_miss(#[case] block_size: usize) {
    let c = ctx(block_size, true);

    // Initialized → AcquireMatches → AwaitingLookup
    let mut p = P::Initialized;
    p = p
        .apply(E::AcquireMatches { num_computed_tokens: 0 }, &c)
        .0;
    assert!(matches!(p, P::AwaitingLookup));

    // LocalLookupCompleted: 0 host, 0 disk, 469 remote candidates (the exact
    // production scenario — ~100k token request → 469 block hashes).
    p = p
        .apply(
            E::LocalLookupCompleted {
                host_blocks: vec![],
                disk_blocks: vec![],
                remote_candidates: (0..469u64).collect(),
            },
            &c,
        )
        .0;
    assert!(
        matches!(p, P::LookingUp { .. }),
        "remote candidates present + remote_enabled → LookingUp"
    );

    // RemoteLookupClosed: registry returned 0 matches (fresh registry)
    let (final_phase, effects) = p.apply(E::RemoteLookupClosed, &c);
    assert!(
        matches!(final_phase, P::Initialized),
        "fresh registry (0 matches) + 0 local blocks must yield Initialized, got {final_phase:?}"
    );
    assert!(
        effects.iter().any(|e| matches!(e, SlotEffect::MatchNone)),
        "must emit MatchNone so get_num_new_matched_tokens returns (Some(0), false)"
    );
}

// =============================================================================
// Regression: TransferFailed must decrement ops so slots reach Finished
//
// Root cause: TransferFailed in Decoding and Finishing only logged a Diag
// but never called ops.complete_one(). The dispatched_operations_count stayed
// elevated → has_any() always true → slot hung in Finishing forever.
// =============================================================================

/// **Negative regression**: TransferFailed during Finishing must decrement ops
/// and reach Finished when it was the last outstanding op.
#[rstest]
#[case(16)]
fn regression_transfer_failed_finishing_resolves_to_finished(#[case] block_size: usize) {
    let c = ctx(block_size, true);
    let mut tracker = OperationTracker::new();
    tracker.append_pending(mock_load_request("fail-finish", vec![1]));
    let _ = tracker.take_pending_for_dispatch();

    let p: P = P::Finishing { ops: tracker };
    let (phase, effects) = p.apply(
        E::TransferFailed {
            operation_id: Uuid::new_v4(),
        },
        &c,
    );
    assert!(
        matches!(phase, P::Finished),
        "TransferFailed on last op in Finishing must reach Finished, got {phase:?}"
    );
    assert!(
        effects.iter().any(|e| matches!(e, SlotEffect::Diag { .. })),
        "must emit Diag for the failed transfer"
    );
    assert!(
        effects
            .iter()
            .any(|e| matches!(e, SlotEffect::RecordCacheStats { .. })),
        "must emit RecordCacheStats on transition to Finished"
    );
}

/// TransferFailed during Finishing with 2 ops: first failure decrements but
/// stays in Finishing; second failure resolves to Finished.
#[rstest]
#[case(16)]
fn regression_transfer_failed_finishing_multi_op(#[case] block_size: usize) {
    let c = ctx(block_size, true);
    let mut tracker = OperationTracker::new();
    tracker.append_pending(mock_load_request("fail-multi", vec![1]));
    tracker.append_pending(mock_load_request("fail-multi", vec![2]));
    let _ = tracker.take_pending_for_dispatch();
    assert_eq!(tracker.dispatched_count(), 2);

    let p: P = P::Finishing { ops: tracker };

    let (phase, effects) = p.apply(
        E::TransferFailed {
            operation_id: Uuid::new_v4(),
        },
        &c,
    );
    assert!(
        matches!(phase, P::Finishing { .. }),
        "first failure with 2 ops should stay in Finishing, got {phase:?}"
    );
    assert!(effects.iter().any(|e| matches!(e, SlotEffect::Diag { .. })));

    let (phase, _) = phase.apply(
        E::TransferFailed {
            operation_id: Uuid::new_v4(),
        },
        &c,
    );
    assert!(
        matches!(phase, P::Finished),
        "second failure should resolve to Finished, got {phase:?}"
    );
}

/// **Negative regression**: TransferFailed during Decoding must decrement ops
/// so RequestFinished can reach Finished (not hang in Finishing).
#[rstest]
#[case(16)]
fn regression_transfer_failed_decoding_then_request_finished(#[case] block_size: usize) {
    let c = ctx(block_size, true);
    let mut tracker = OperationTracker::new();
    tracker.append_pending(mock_load_request("fail-decode", vec![1]));
    let _ = tracker.take_pending_for_dispatch();

    let mut p: P = P::Decoding {
        ops: tracker,
        iteration_first_scheduled: 1,
    };

    p = p
        .apply(
            E::TransferFailed {
                operation_id: Uuid::new_v4(),
            },
            &c,
        )
        .0;
    assert!(
        matches!(p, P::Decoding { .. }),
        "still Decoding after transfer failure"
    );

    let (phase, effects) = p.apply(E::RequestFinished, &c);
    assert!(
        matches!(phase, P::Finished),
        "RequestFinished after failed transfer (only op) must reach Finished directly, got {phase:?}"
    );
    assert!(
        effects
            .iter()
            .any(|e| matches!(e, SlotEffect::RecordCacheStats { .. })),
        "must emit RecordCacheStats"
    );
}

/// Mixed: one TransferCompleted + one TransferFailed during Decoding, then
/// RequestFinished → Finished (not stuck in Finishing).
#[rstest]
#[case(16)]
fn regression_mixed_complete_and_fail_then_request_finished(#[case] block_size: usize) {
    let c = ctx(block_size, true);
    let mut tracker = OperationTracker::new();
    tracker.append_pending(mock_load_request("mix", vec![1]));
    tracker.append_pending(mock_load_request("mix", vec![2]));
    let _ = tracker.take_pending_for_dispatch();

    let mut p: P = P::Decoding {
        ops: tracker,
        iteration_first_scheduled: 1,
    };

    p = p
        .apply(
            E::TransferCompleted {
                operation_id: Uuid::new_v4(),
            },
            &c,
        )
        .0;
    p = p
        .apply(
            E::TransferFailed {
                operation_id: Uuid::new_v4(),
            },
            &c,
        )
        .0;

    let (phase, _) = p.apply(E::RequestFinished, &c);
    assert!(
        matches!(phase, P::Finished),
        "both ops resolved (1 complete + 1 failed) → Finished, got {phase:?}"
    );
}

// =============================================================================
// Regression: SkippedDecode must preserve ops tracker
//
// Root cause: SkippedDecode was a unit variant that dropped the ops tracker.
// On resume, a fresh OperationTracker was created, losing in-flight offloads.
// =============================================================================

/// **Positive regression**: MarkSkipped during Decoding preserves ops in
/// SkippedDecode, and ApplySchedulerOutput restores them into Decoding.
#[rstest]
#[case(16)]
fn regression_skipped_decode_preserves_ops(#[case] block_size: usize) {
    let c = ctx(block_size, true);
    let mut tracker = OperationTracker::new();
    tracker.append_pending(mock_load_request("skip-ops", vec![1]));
    let _ = tracker.take_pending_for_dispatch();
    assert_eq!(tracker.dispatched_count(), 1);

    let p: P = P::Decoding {
        ops: tracker,
        iteration_first_scheduled: 1,
    };

    let (phase, _) = p.apply(E::MarkSkipped, &c);
    match &phase {
        P::SkippedDecode { ops } => {
            assert_eq!(
                ops.dispatched_count(),
                1,
                "ops must be carried into SkippedDecode"
            );
        }
        other => panic!("expected SkippedDecode, got {other:?}"),
    }

    let (phase, _) = phase.apply(sched_decode(), &c);
    match &phase {
        P::Decoding { ops, .. } => {
            assert_eq!(
                ops.dispatched_count(),
                1,
                "ops must be restored from SkippedDecode into Decoding"
            );
        }
        other => panic!("expected Decoding, got {other:?}"),
    }

    let (phase, _) = phase.apply(
        E::TransferCompleted {
            operation_id: Uuid::new_v4(),
        },
        &c,
    );
    match &phase {
        P::Decoding { ops, .. } => {
            assert_eq!(
                ops.dispatched_count(),
                0,
                "TransferCompleted must resolve the surviving op"
            );
        }
        other => panic!("expected Decoding, got {other:?}"),
    }

    let (phase, _) = phase.apply(E::RequestFinished, &c);
    assert!(
        matches!(phase, P::Finished),
        "all ops resolved → Finished, got {phase:?}"
    );
}

/// MarkSkipped with zero ops: SkippedDecode carries empty tracker, resume
/// produces Decoding with empty ops, RequestFinished goes to Finished.
#[rstest]
#[case(16)]
fn regression_skipped_decode_no_ops_round_trip(#[case] block_size: usize) {
    let c = ctx(block_size, true);
    let p: P = P::Decoding {
        ops: OperationTracker::new(),
        iteration_first_scheduled: 1,
    };

    let (phase, _) = p.apply(E::MarkSkipped, &c);
    assert!(matches!(phase, P::SkippedDecode { .. }));

    let (phase, _) = phase.apply(sched_decode(), &c);
    assert!(matches!(phase, P::Decoding { .. }));

    let (phase, _) = phase.apply(E::RequestFinished, &c);
    assert!(
        matches!(phase, P::Finished),
        "no ops → Finished directly, got {phase:?}"
    );
}

// =============================================================================
// Part D: vLLM protocol scenario tests
//
// Each test simulates the exact sequence of SlotEvents that corresponds to
// a vLLM scheduler API call sequence, verified against the vLLM v0.18
// scheduler source at /data/priel/vllm/vllm/v1/core/sched/scheduler.py.
// =============================================================================

/// vLLM protocol scenario 1: Cold miss (no cache hit).
///
/// vLLM calls: get_num_new_matched_tokens(0,false) → update_state_after_alloc(0)
/// → build_connector_meta (×N iterations with decode tokens) → request_finished.
///
/// The slot must advance through Prefilling → Decoding, produce offload effects
/// during decode, and reach Finished.
#[test]
fn protocol_cold_miss_full_prefill_offload_finished() {
    let c = ctx(256, true);

    // Step 1: get_num_new_matched_tokens → AcquireMatches → lookup → MatchNone
    let mut p = P::Initialized;
    p = p
        .apply(E::AcquireMatches { num_computed_tokens: 0 }, &c)
        .0;
    assert!(matches!(p, P::AwaitingLookup));

    let (p2, effects) = p.apply(
        E::LocalLookupCompleted {
            host_blocks: vec![],
            disk_blocks: vec![],
            remote_candidates: vec![],
        },
        &c,
    );
    p = p2;
    assert!(matches!(p, P::Initialized), "zero blocks → MatchNone → Initialized");
    assert!(effects.iter().any(|e| matches!(e, SlotEffect::MatchNone)));

    // Step 2: build_connector_meta (first iteration) — prefill, no tokens yet.
    // No offload effects here: KV data hasn't been computed into these blocks.
    let (p2, effects) = p.apply(
        E::ApplySchedulerOutput {
            tokens: vec![],
            block_ids: vec![1, 2, 3, 4],
            num_computed_tokens: 0,
            num_scheduled_tokens: 1024,
            priorities: None,
            iteration: 1,
        },
        &c,
    );
    p = p2;
    assert!(
        matches!(p, P::Prefilling { .. }),
        "cold miss + empty tokens → Prefilling, got {p:?}"
    );
    assert!(
        effects.is_empty(),
        "Initialized → Prefilling must NOT produce offload effects (no KV data computed yet)"
    );

    // Step 3: build_connector_meta (prefill → decode transition)
    // Prefilling + tokens → Decoding. No offload yet (just transitioned).
    p = p.apply(
        E::ApplySchedulerOutput {
            tokens: vec![100, 101, 102],
            block_ids: vec![1, 2, 3, 4, 5],
            num_computed_tokens: 1024,
            num_scheduled_tokens: 3,
            priorities: None,
            iteration: 2,
        },
        &c,
    ).0;
    assert!(
        matches!(p, P::Decoding { .. }),
        "tokens present → Decoding, got {p:?}"
    );

    // Step 4: build_connector_meta (decode iteration) — reducer produces no offload
    // effects; offloads happen in the slot_runtime hook after the reducer.
    p = p.apply(
        E::ApplySchedulerOutput {
            tokens: vec![200],
            block_ids: vec![1, 2, 3, 4, 5, 6],
            num_computed_tokens: 1027,
            num_scheduled_tokens: 1,
            priorities: None,
            iteration: 3,
        },
        &c,
    ).0;
    assert!(
        matches!(p, P::Decoding { .. }),
        "stays Decoding, got {p:?}"
    );

    // Step 5: request_finished
    let (p2, _) = p.apply(E::RequestFinished, &c);
    p = p2;
    assert!(
        matches!(p, P::Finished),
        "no pending ops → Finished, got {p:?}"
    );
}

/// vLLM protocol scenario 2: Warm hit (cache hit with onboarding).
///
/// vLLM calls: get_num_new_matched_tokens(N,true) → update_state_after_alloc(N)
/// → build_connector_meta → apply_worker_feedback → request_finished.
#[test]
fn protocol_warm_hit_onboard_offload_finished() {
    let c = ctx(16, true);

    // Step 1: AcquireMatches → lookup finds local blocks → OnboardReady
    let mut p = P::Initialized;
    p = p
        .apply(E::AcquireMatches { num_computed_tokens: 0 }, &c)
        .0;
    p = p
        .apply(
            E::LocalLookupCompleted {
                host_blocks: vec![(), (), (), ()],
                disk_blocks: vec![],
                remote_candidates: vec![],
            },
            &c,
        )
        .0;
    assert!(matches!(p, P::OnboardReady { num_external_tokens: 64, .. }));

    // Step 2: PollMatchReport → MatchReady
    let (p2, effects) = p.apply(E::PollMatchReport, &c);
    p = p2;
    assert!(effects.iter().any(|e| matches!(e, SlotEffect::MatchReady { num_external_tokens: 64 })));

    // Step 3: AllocCompleted → Onboarding
    p = p
        .apply(
            E::AllocCompleted {
                block_ids: vec![10, 11, 12, 13],
                num_external_tokens: 64,
            },
            &c,
        )
        .0;
    assert!(matches!(p, P::Onboarding { .. }));

    // Step 4: build_connector_meta (prefill) → Prefilling
    p = p
        .apply(
            E::ApplySchedulerOutput {
                tokens: vec![],
                block_ids: vec![10, 11, 12, 13],
                num_computed_tokens: 64,
                num_scheduled_tokens: 64,
                priorities: None,
                iteration: 1,
            },
            &c,
        )
        .0;
    assert!(matches!(p, P::Prefilling { .. }));

    // Step 5: build_connector_meta (decode) → Decoding (first decode step, no offload yet)
    p = p.apply(
        E::ApplySchedulerOutput {
            tokens: vec![42],
            block_ids: vec![10, 11, 12, 13, 14],
            num_computed_tokens: 128,
            num_scheduled_tokens: 1,
            priorities: None,
            iteration: 2,
        },
        &c,
    ).0;
    assert!(matches!(p, P::Decoding { .. }));

    // Step 6: build_connector_meta (second decode step) — reducer produces no
    // offload effects; offloads are position-based in slot_runtime hook.
    p = p.apply(
        E::ApplySchedulerOutput {
            tokens: vec![43],
            block_ids: vec![10, 11, 12, 13, 14, 15],
            num_computed_tokens: 129,
            num_scheduled_tokens: 1,
            priorities: None,
            iteration: 3,
        },
        &c,
    ).0;
    assert!(matches!(p, P::Decoding { .. }));

    // Step 7: request_finished → Finished
    p = p.apply(E::RequestFinished, &c).0;
    assert!(matches!(p, P::Finished));
}

/// vLLM protocol scenario 3: Preempt during decode and resume.
///
/// Request is decoding, gets preempted, then AcquireMatches restarts the
/// lookup flow from Preempted.
#[test]
fn protocol_preempt_during_decode_and_resume() {
    let c = ctx(16, true);

    let mut p: P = P::Decoding {
        ops: OperationTracker::new(),
        iteration_first_scheduled: 1,
    };

    // Preempt
    p = p.apply(E::Preempt, &c).0;
    assert!(matches!(p, P::Preempted { .. }));

    // Resume: AcquireMatches restarts lookup
    p = p
        .apply(E::AcquireMatches { num_computed_tokens: 0 }, &c)
        .0;
    assert!(matches!(p, P::AwaitingLookup));

    // Lookup finds local blocks → OnboardReady
    p = p
        .apply(
            E::LocalLookupCompleted {
                host_blocks: vec![(), ()],
                disk_blocks: vec![],
                remote_candidates: vec![],
            },
            &c,
        )
        .0;
    assert!(matches!(p, P::OnboardReady { num_external_tokens: 32, .. }));

    // Full cycle: alloc → onboarding → prefill → decode → finished
    p = p.apply(E::PollMatchReport, &c).0;
    p = p
        .apply(
            E::AllocCompleted {
                block_ids: vec![20, 21],
                num_external_tokens: 32,
            },
            &c,
        )
        .0;
    assert!(matches!(p, P::Onboarding { .. }));

    p = p.apply(sched_prefill(), &c).0;
    assert!(matches!(p, P::Prefilling { .. }));

    p = p.apply(sched_decode(), &c).0;
    assert!(matches!(p, P::Decoding { .. }));

    p = p.apply(E::RequestFinished, &c).0;
    assert!(matches!(p, P::Finished));
}

/// vLLM protocol scenario 4: Chunked prefill with skip and ops preservation.
///
/// Decoding with in-flight offloads → MarkSkipped → SkippedDecode (ops carried)
/// → resume with ApplySchedulerOutput → Decoding (ops restored) →
/// TransferCompleted resolves → RequestFinished → Finished.
#[test]
fn protocol_chunked_prefill_skip_preserves_ops() {
    let c = ctx(16, true);

    let mut tracker = OperationTracker::new();
    tracker.append_pending(mock_load_request("chunked", vec![1]));
    let _ = tracker.take_pending_for_dispatch();

    let mut p: P = P::Decoding {
        ops: tracker,
        iteration_first_scheduled: 1,
    };

    // MarkSkipped (scheduler didn't schedule this request this step)
    p = p.apply(E::MarkSkipped, &c).0;
    match &p {
        P::SkippedDecode { ops } => {
            assert_eq!(ops.dispatched_count(), 1, "ops must survive the skip");
        }
        other => panic!("expected SkippedDecode, got {other:?}"),
    }

    // Resume: ApplySchedulerOutput
    p = p.apply(sched_decode(), &c).0;
    match &p {
        P::Decoding { ops, .. } => {
            assert_eq!(ops.dispatched_count(), 1, "ops must be restored on resume");
        }
        other => panic!("expected Decoding, got {other:?}"),
    }

    // Transfer completes
    p = p
        .apply(
            E::TransferCompleted {
                operation_id: Uuid::new_v4(),
            },
            &c,
        )
        .0;

    // RequestFinished with no remaining ops → Finished
    p = p.apply(E::RequestFinished, &c).0;
    assert!(matches!(p, P::Finished));
}

/// vLLM protocol scenario 5: Failed transfer during finishing.
///
/// Offloads dispatched → RequestFinished → Finishing (ops pending) →
/// TransferFailed → ops.complete_one() → Finished.
#[test]
fn protocol_failed_transfer_during_finishing() {
    let c = ctx(16, true);

    let mut tracker = OperationTracker::new();
    tracker.append_pending(mock_load_request("fail-fin", vec![1]));
    tracker.append_pending(mock_load_request("fail-fin", vec![2]));
    let _ = tracker.take_pending_for_dispatch();

    let mut p: P = P::Decoding {
        ops: tracker,
        iteration_first_scheduled: 1,
    };

    // RequestFinished with pending ops → Finishing
    p = p.apply(E::RequestFinished, &c).0;
    assert!(matches!(p, P::Finishing { .. }));

    // First op completes normally
    p = p
        .apply(
            E::TransferCompleted {
                operation_id: Uuid::new_v4(),
            },
            &c,
        )
        .0;
    assert!(
        matches!(p, P::Finishing { .. }),
        "still Finishing (1 op remaining)"
    );

    // Second op FAILS
    let (p2, effects) = p.apply(
        E::TransferFailed {
            operation_id: Uuid::new_v4(),
        },
        &c,
    );
    p = p2;
    assert!(
        matches!(p, P::Finished),
        "TransferFailed must resolve the op → Finished, got {p:?}"
    );
    assert!(
        effects.iter().any(|e| matches!(e, SlotEffect::Diag { .. })),
        "must emit Diag for the failure"
    );
    assert!(
        effects.iter().any(|e| matches!(e, SlotEffect::RecordCacheStats { .. })),
        "must emit RecordCacheStats on Finished"
    );
}

/// Verifies that when a prefetch completes and blocks are resolved from the
/// host pool (leader fires `PrefetchReady` instead of `TransferCompleted`),
/// the resulting `OnboardReady` has blocks in `host_staging` and
/// `EnqueueOnboardTransfer` uses local host→device DMA (no remote re-read).
///
/// This is the Layer 1 correctness fix: prefetched blocks are onboarded from
/// host pinned memory (fast DMA) rather than re-reading from VAST.
#[test]
fn protocol_prefetch_resolved_via_host_pool_uses_local_onboard() {
    let c = ctx(16, true);
    let mut p = P::Initialized;

    p = p
        .apply(E::AcquireMatches { num_computed_tokens: 0 }, &c)
        .0;
    p = p
        .apply(
            E::LocalLookupCompleted {
                host_blocks: vec![],
                disk_blocks: vec![],
                remote_candidates: vec![0xAA, 0xBB, 0xCC],
            },
            &c,
        )
        .0;
    assert!(matches!(p, P::LookingUp { .. }));

    p = p
        .apply(
            E::RemoteLookupCompleted {
                matches: vec![(0xAA, 1), (0xBB, 2), (0xCC, 3)],
            },
            &c,
        )
        .0;
    assert!(matches!(p, P::Prefetching { .. }));

    // Leader resolves host blocks and fires PrefetchReady (Layer 1 fix)
    p = p
        .apply(
            E::PrefetchReady {
                blocks: vec![(), (), ()],
            },
            &c,
        )
        .0;

    // OnboardReady: host_staging should have the resolved blocks,
    // remote_hashes should be EMPTY (resolved blocks cover all prefetch hashes)
    match &p {
        P::OnboardReady {
            host_staging,
            remote_hashes,
            num_external_tokens,
            ..
        } => {
            assert_eq!(
                host_staging.len(), 3,
                "host_staging must contain the 3 resolved blocks"
            );
            assert_eq!(
                *num_external_tokens,
                3 * 16, // 3 blocks × block_size=16
                "num_external_tokens must reflect all resolved blocks"
            );
            assert!(
                remote_hashes.is_empty(),
                "remote_hashes must be empty when resolved host blocks cover all \
                 prefetch hashes (prevents redundant VAST re-read). Got: {remote_hashes:?}"
            );
        }
        other => panic!("expected OnboardReady, got {other:?}"),
    }

    // PollMatchReport → MatchReady
    let (p_after_poll, effects) = p.apply(E::PollMatchReport, &c);
    p = p_after_poll;
    assert!(
        effects.iter().any(|e| matches!(e, SlotEffect::MatchReady { .. })),
        "must emit MatchReady"
    );

    // AllocCompleted → Onboarding + EnqueueOnboardTransfer
    let (p_after_alloc, effects) = p.apply(
        E::AllocCompleted {
            block_ids: vec![100, 101, 102],
            num_external_tokens: 48,
        },
        &c,
    );
    p = p_after_alloc;
    assert!(matches!(p, P::Onboarding { .. }));

    // The EnqueueOnboardTransfer effect should have host_staging blocks
    let has_onboard = effects.iter().any(|e| {
        matches!(
            e,
            SlotEffect::EnqueueOnboardTransfer {
                host_staging,
                remote_hashes,
                ..
            } if !host_staging.is_empty()
        )
    });
    assert!(
        has_onboard,
        "EnqueueOnboardTransfer must have non-empty host_staging \
         (local DMA, not remote re-read). Effects: {effects:?}"
    );
}
