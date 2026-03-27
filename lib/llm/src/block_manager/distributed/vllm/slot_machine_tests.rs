// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Comprehensive unit tests for the `RequestPhase::apply()` reducer.
//!
//! Covers every row in the slot state-machine transition table from
//! `docs/slot-state-machine-design.md`, plus late-event no-ops, edge cases,
//! and `SlotState` compatibility views.
//!
//! Uses `()` for both `H` and `D` type parameters -- the reducer's
//! logic is independent of the concrete block type.

#![cfg(test)]

use super::slot_machine::*;
use super::test_harness::helpers::{mock_load_request, tracker_with_dispatched};
use super::OperationTracker;

// Remote match types simplified to (u64, u64) in slot_machine.rs
use crate::block_manager::distributed::vllm::SlotState;

use rstest::rstest;
use uuid::Uuid;

// ═══════════════════════════════════════════════════════════════════════════
// Type aliases — unit-type blocks simplify test construction
// ═══════════════════════════════════════════════════════════════════════════

type TestPhase = RequestPhase<(), ()>;
type TestEvent = SlotEvent<(), ()>;

#[allow(dead_code)]
type TestEffect = SlotEffect<(), ()>;

// ═══════════════════════════════════════════════════════════════════════════
// Test fixtures
// ═══════════════════════════════════════════════════════════════════════════

fn mock_ctx() -> SlotContext {
    SlotContext {
        block_size: 16,
        remote_enabled: true,
        g4_xfer_fail_policy: G4FailPolicy::Fallback,
    }
}

fn mock_ctx_abort() -> SlotContext {
    SlotContext {
        block_size: 16,
        remote_enabled: true,
        g4_xfer_fail_policy: G4FailPolicy::Abort,
    }
}

fn mock_ctx_no_remote() -> SlotContext {
    SlotContext {
        block_size: 16,
        remote_enabled: false,
        g4_xfer_fail_policy: G4FailPolicy::Fallback,
    }
}

fn mock_prefetch() -> PrefetchState {
    PrefetchState {
        operation_id: Uuid::new_v4(),
        sequence_hashes: vec![0x1234, 0x5678],
        num_external_tokens: 256,
        started_at: std::time::Instant::now(),
    }
}

fn mock_remote_match() -> (u64, u64) {
    (0xABCD, 0x1234)
}

fn make_scheduler_output_prefill() -> TestEvent {
    SlotEvent::ApplySchedulerOutput {
        tokens: vec![],
        block_ids: vec![1, 2],
        num_computed_tokens: 0,
        num_scheduled_tokens: 128,
        priorities: None,
        iteration: 1,
    }
}

fn make_scheduler_output_decode() -> TestEvent {
    SlotEvent::ApplySchedulerOutput {
        tokens: vec![42, 43, 44],
        block_ids: vec![1, 2, 3],
        num_computed_tokens: 128,
        num_scheduled_tokens: 1,
        priorities: Some(vec![10, 20, 30]),
        iteration: 1,
    }
}

fn onboard_ready_pending(n_host: usize, n_disk: usize, n_tokens: usize) -> TestPhase {
    RequestPhase::OnboardReady {
        num_external_tokens: n_tokens,
        host_staging: vec![(); n_host],
        disk_staging: vec![(); n_disk],
        remote_hashes: vec![0x1111; n_host],
        disclosure: DisclosureState::Pending,
        prefetched_blocks_for_stats: 0,
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Lookup transitions
// Transition table rows:
//   (Initialized, AcquireMatches) -> emit RunLocalLookup -> AwaitingLookup
//   (AwaitingLookup, LocalLookupCompleted) -> 3 branches
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_initialized_acquire_matches() {
    // (Initialized, AcquireMatches) -> emit RunLocalLookup -> AwaitingLookup
    let phase: TestPhase = RequestPhase::Initialized;
    let (new_phase, effects) = phase.apply(
        SlotEvent::AcquireMatches {
            num_computed_tokens: 0,
        },
        &mock_ctx(),
    );
    assert!(matches!(new_phase, RequestPhase::AwaitingLookup));
    assert_eq!(effects.len(), 1);
    assert!(matches!(
        effects[0],
        SlotEffect::RunLocalLookup {
            num_computed_tokens: 0
        }
    ));
}

#[test]
fn test_awaiting_lookup_completed_with_remote() {
    // (AwaitingLookup, LocalLookupCompleted { remote_candidates non-empty, remote_enabled })
    // -> emit StartRemoteLookup -> LookingUp { host, disk }
    let phase: TestPhase = RequestPhase::AwaitingLookup;
    let (new_phase, effects) = phase.apply(
        SlotEvent::LocalLookupCompleted {
            host_blocks: vec![(), ()],
            disk_blocks: vec![()],
            remote_candidates: vec![0xAAAA, 0xBBBB],
        },
        &mock_ctx(),
    );

    assert!(matches!(new_phase, RequestPhase::LookingUp { .. }));
    if let RequestPhase::LookingUp {
        host_blocks,
        disk_blocks,
    } = &new_phase
    {
        assert_eq!(host_blocks.len(), 2);
        assert_eq!(disk_blocks.len(), 1);
    }

    assert!(effects
        .iter()
        .any(|e| matches!(e, SlotEffect::StartRemoteLookup { .. })));
    if let Some(SlotEffect::StartRemoteLookup {
        candidates,
        host_blocks,
        disk_blocks,
    }) = effects
        .iter()
        .find(|e| matches!(e, SlotEffect::StartRemoteLookup { .. }))
    {
        assert_eq!(candidates, &[0xAAAA, 0xBBBB]);
        // Staging blocks are moved into `LookingUp` only; the effect carries candidates
        // for the executor (logging uses counts from the phase after apply_and_execute).
        assert!(host_blocks.is_empty());
        assert!(disk_blocks.is_empty());
    }
}

#[test]
fn test_awaiting_lookup_completed_local_only() {
    // (AwaitingLookup, LocalLookupCompleted { host+disk, no remote })
    // -> OnboardReady { disclosure: Pending }
    let phase: TestPhase = RequestPhase::AwaitingLookup;
    let (new_phase, effects) = phase.apply(
        SlotEvent::LocalLookupCompleted {
            host_blocks: vec![(), (), ()],
            disk_blocks: vec![()],
            remote_candidates: vec![],
        },
        &mock_ctx(),
    );

    assert!(matches!(new_phase, RequestPhase::OnboardReady { .. }));
    if let RequestPhase::OnboardReady {
        host_staging,
        disk_staging,
        disclosure,
        ..
    } = &new_phase
    {
        assert_eq!(host_staging.len(), 3);
        assert_eq!(disk_staging.len(), 1);
        assert_eq!(*disclosure, DisclosureState::Pending);
    }
    assert!(!effects
        .iter()
        .any(|e| matches!(e, SlotEffect::StartRemoteLookup { .. })));
}

#[test]
fn test_awaiting_lookup_completed_nothing_matched() {
    // (AwaitingLookup, LocalLookupCompleted { nothing matched })
    // -> emit MatchNone -> Initialized
    let phase: TestPhase = RequestPhase::AwaitingLookup;
    let (new_phase, effects) = phase.apply(
        SlotEvent::LocalLookupCompleted {
            host_blocks: vec![],
            disk_blocks: vec![],
            remote_candidates: vec![],
        },
        &mock_ctx(),
    );

    assert!(matches!(new_phase, RequestPhase::Initialized));
    assert!(effects
        .iter()
        .any(|e| matches!(e, SlotEffect::MatchNone)));
}

#[test]
fn test_awaiting_lookup_remote_disabled_falls_back_to_local() {
    // Edge case: remote_candidates present but remote_enabled=false
    // -> should treat like local-only -> OnboardReady { Pending }
    let phase: TestPhase = RequestPhase::AwaitingLookup;
    let (new_phase, effects) = phase.apply(
        SlotEvent::LocalLookupCompleted {
            host_blocks: vec![()],
            disk_blocks: vec![],
            remote_candidates: vec![0xDEAD, 0xBEEF],
        },
        &mock_ctx_no_remote(),
    );

    assert!(matches!(new_phase, RequestPhase::OnboardReady { .. }));
    assert!(!effects
        .iter()
        .any(|e| matches!(e, SlotEffect::StartRemoteLookup { .. })));
}

// ═══════════════════════════════════════════════════════════════════════════
// Remote lookup transitions
// Transition table rows:
//   (LookingUp, RemoteLookupCompleted) -> merge, StartPrefetch or OnboardReady
//   (LookingUp, RemoteLookupClosed) -> OnboardReady with host+disk only
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_looking_up_remote_completed_with_hashes() {
    // (LookingUp, RemoteLookupCompleted { matches with hashes })
    // -> emit StartPrefetch -> Prefetching
    let phase: TestPhase = RequestPhase::LookingUp {
        host_blocks: vec![(), ()],
        disk_blocks: vec![()],
    };
    let (new_phase, effects) = phase.apply(
        SlotEvent::RemoteLookupCompleted {
            matches: vec![mock_remote_match(), mock_remote_match()],
        },
        &mock_ctx(),
    );

    assert!(matches!(new_phase, RequestPhase::Prefetching { .. }));
    if let RequestPhase::Prefetching {
        host_staging,
        disk_staging,
        ..
    } = &new_phase
    {
        assert_eq!(host_staging.len(), 2);
        assert_eq!(disk_staging.len(), 1);
    }
    assert!(effects
        .iter()
        .any(|e| matches!(e, SlotEffect::StartPrefetch { .. })));
}

#[test]
fn test_looking_up_remote_completed_no_hashes() {
    // (LookingUp, RemoteLookupCompleted { no matches })
    // -> OnboardReady { disclosure: Pending }
    let phase: TestPhase = RequestPhase::LookingUp {
        host_blocks: vec![(), ()],
        disk_blocks: vec![()],
    };
    let (new_phase, _effects) = phase.apply(
        SlotEvent::RemoteLookupCompleted { matches: vec![] },
        &mock_ctx(),
    );

    assert!(matches!(new_phase, RequestPhase::OnboardReady { .. }));
    if let RequestPhase::OnboardReady {
        host_staging,
        disk_staging,
        disclosure,
        ..
    } = &new_phase
    {
        assert_eq!(host_staging.len(), 2);
        assert_eq!(disk_staging.len(), 1);
        assert_eq!(*disclosure, DisclosureState::Pending);
    }
}

#[test]
fn test_looking_up_remote_closed() {
    // (LookingUp, RemoteLookupClosed) -> OnboardReady with host+disk only
    let phase: TestPhase = RequestPhase::LookingUp {
        host_blocks: vec![(), (), ()],
        disk_blocks: vec![()],
    };
    let (new_phase, _effects) = phase.apply(SlotEvent::RemoteLookupClosed, &mock_ctx());

    assert!(matches!(new_phase, RequestPhase::OnboardReady { .. }));
    if let RequestPhase::OnboardReady {
        host_staging,
        disk_staging,
        remote_hashes,
        disclosure,
        ..
    } = &new_phase
    {
        assert_eq!(host_staging.len(), 3);
        assert_eq!(disk_staging.len(), 1);
        assert!(remote_hashes.is_empty(), "no remote when lookup closed");
        assert_eq!(*disclosure, DisclosureState::Pending);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Prefetch transitions
// Transition table rows:
//   (Prefetching, PrefetchReady)   -> extend host_staging -> OnboardReady
//   (Prefetching, PrefetchTimeout) -> staging dropped -> Preempted { false }
//   (Prefetching, PrefetchFailed)  -> Fallback w/ staging -> OnboardReady
//   (Prefetching, PrefetchFailed)  -> Fallback no staging -> Preempted { true }
//   (Prefetching, PrefetchFailed)  -> Abort policy -> Preempted { true }
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_prefetching_ready() {
    // (Prefetching, PrefetchReady { blocks })
    // -> extend host_staging with blocks -> OnboardReady { Pending }
    let phase: TestPhase = RequestPhase::Prefetching {
        prefetch: mock_prefetch(),
        host_staging: vec![(), ()],
        disk_staging: vec![()],
        prefetched_blocks_for_stats: 2,
    };
    let (new_phase, _effects) = phase.apply(
        SlotEvent::PrefetchReady {
            blocks: vec![(), ()],
        },
        &mock_ctx(),
    );

    assert!(matches!(new_phase, RequestPhase::OnboardReady { .. }));
    if let RequestPhase::OnboardReady {
        host_staging,
        disk_staging,
        disclosure,
        prefetched_blocks_for_stats,
        ..
    } = &new_phase
    {
        assert_eq!(host_staging.len(), 4, "2 original + 2 prefetched");
        assert_eq!(disk_staging.len(), 1);
        assert_eq!(*disclosure, DisclosureState::Pending);
        assert_eq!(*prefetched_blocks_for_stats, 2);
    }
}

#[test]
fn test_prefetching_ready_with_empty_blocks() {
    // Edge case: PrefetchReady with no blocks — still transitions to OnboardReady
    let phase: TestPhase = RequestPhase::Prefetching {
        prefetch: mock_prefetch(),
        host_staging: vec![(), ()],
        disk_staging: vec![],
        prefetched_blocks_for_stats: 0,
    };
    let (new_phase, _effects) =
        phase.apply(SlotEvent::PrefetchReady { blocks: vec![] }, &mock_ctx());

    assert!(matches!(new_phase, RequestPhase::OnboardReady { .. }));
    if let RequestPhase::OnboardReady {
        host_staging,
        disk_staging,
        ..
    } = &new_phase
    {
        assert_eq!(host_staging.len(), 2, "unchanged when no blocks prefetched");
        assert!(disk_staging.is_empty());
    }
}

#[test]
fn test_prefetching_timeout() {
    // (Prefetching, PrefetchTimeout)
    // -> staging MOVED OUT and DROPPED -> Preempted { recovered_from_failure: false }
    let phase: TestPhase = RequestPhase::Prefetching {
        prefetch: mock_prefetch(),
        host_staging: vec![(), (), ()],
        disk_staging: vec![(), ()],
        prefetched_blocks_for_stats: 3,
    };
    let (new_phase, effects) = phase.apply(SlotEvent::PrefetchTimeout, &mock_ctx());

    assert!(matches!(
        new_phase,
        RequestPhase::Preempted {
            recovered_from_failure: false
        }
    ));
    // Staging was consumed by `apply()` — ownership transfer guarantees drop.
    // May emit ReleaseStaging to notify coordinator.
    assert!(
        effects.is_empty()
            || effects
                .iter()
                .all(|e| matches!(e, SlotEffect::ReleaseStaging | SlotEffect::Diag { .. }))
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// G4 host-prefetch timeout race (distributed disk / slow R2H)
//
// Production (`EngineDeadError`): poll path fired a short prefetch wait timeout
// while chunked R2H was still finishing; staging was torn down, then completion
// signals arrived late and vLLM's block table diverged from KVBM.
//
// These tests pin the reducer behavior for that ordering so late events after
// `PrefetchTimeout` cannot accidentally revive `OnboardReady`.
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_g4_prefetch_timeout_then_late_prefetch_ready_stays_preempted() {
    let ctx = mock_ctx();
    let phase = RequestPhase::Prefetching {
        prefetch: mock_prefetch(),
        host_staging: vec![(); 3],
        disk_staging: vec![(); 2],
        prefetched_blocks_for_stats: 3,
    };
    let (phase, _) = phase.apply(SlotEvent::PrefetchTimeout, &ctx);
    assert!(matches!(
        phase,
        RequestPhase::Preempted {
            recovered_from_failure: false
        }
    ));

    let (phase, effects) = phase.apply(
        SlotEvent::PrefetchReady {
            blocks: vec![(), ()],
        },
        &ctx,
    );
    assert!(matches!(
        phase,
        RequestPhase::Preempted {
            recovered_from_failure: false
        }
    ));
    assert!(
        effects.is_empty(),
        "late PrefetchReady after timeout must not apply (would desync engine)"
    );
}

#[test]
fn test_g4_prefetch_timeout_then_late_transfer_completed_stays_preempted() {
    let ctx = mock_ctx();
    let op_id = Uuid::new_v4();
    let prefetch = PrefetchState {
        operation_id: op_id,
        sequence_hashes: vec![0xAAAA, 0xBBBB],
        num_external_tokens: 256,
        started_at: std::time::Instant::now(),
    };
    let phase: TestPhase = RequestPhase::Prefetching {
        prefetch,
        host_staging: vec![(); 2],
        disk_staging: vec![],
        prefetched_blocks_for_stats: 2,
    };
    let (phase, _) = phase.apply(SlotEvent::PrefetchTimeout, &ctx);
    let (phase, effects) = phase.apply(SlotEvent::TransferCompleted { operation_id: op_id }, &ctx);
    assert!(matches!(
        phase,
        RequestPhase::Preempted {
            recovered_from_failure: false
        }
    ));
    assert!(
        effects.is_empty(),
        "late TransferCompleted after prefetch timeout must not mutate phase"
    );
}

#[test]
fn test_g4_prefetch_race_full_sequence_retry_after_timeout() {
    let ctx = mock_ctx();
    let phase = RequestPhase::LookingUp {
        host_blocks: vec![(), ()],
        disk_blocks: vec![()],
    };
    let (phase, _) = phase.apply(
        SlotEvent::RemoteLookupCompleted {
            matches: vec![mock_remote_match()],
        },
        &ctx,
    );
    assert!(matches!(phase, RequestPhase::Prefetching { .. }));

    let (phase, _) = phase.apply(SlotEvent::PrefetchTimeout, &ctx);
    assert!(matches!(
        phase,
        RequestPhase::Preempted {
            recovered_from_failure: false
        }
    ));

    let (phase, _) = phase.apply(
        SlotEvent::PrefetchReady {
            blocks: vec![(), (), ()],
        },
        &ctx,
    );
    assert!(matches!(
        phase,
        RequestPhase::Preempted {
            recovered_from_failure: false
        }
    ));

    let (phase, effects) = phase.apply(
        SlotEvent::AcquireMatches {
            num_computed_tokens: 0,
        },
        &ctx,
    );
    assert!(matches!(phase, RequestPhase::AwaitingLookup));
    assert!(
        effects
            .iter()
            .any(|e| matches!(e, SlotEffect::RunLocalLookup { num_computed_tokens: 0 })),
        "after G4 prefetch timeout, AcquireMatches should restart local lookup"
    );
}

/// Reducer-only simulation of the `error.log` / EngineDeadError timeline (no vLLM).
///
/// Matches container `180b9a534d37`: remote lookup → `Prefetching`, 4s poll wall fires
/// `PrefetchTimeout` → `Preempted` while chunked R2H is still finishing; late
/// `TransferCompleted` / `PrefetchReady` must not resurrect staging; then
/// `AcquireMatches` restarts lookup as after "ZERO matched (full prefill)".
///
/// Name uses `test_g4_prefetch_*` so `cargo test test_g4_prefetch` picks this up with the
/// other G4 prefetch race tests (`|` is not supported as a filter alternation).
#[test]
fn test_g4_prefetch_error_log_timeline_reducer_only() {
    let ctx = mock_ctx();

    // 1) Remote registry hit → StartPrefetch / Prefetching (G4 host prefetch pipeline running)
    let phase = RequestPhase::LookingUp {
        host_blocks: vec![(), ()],
        disk_blocks: vec![()],
    };
    let (phase, effects) = phase.apply(
        SlotEvent::RemoteLookupCompleted {
            matches: vec![mock_remote_match()],
        },
        &ctx,
    );
    assert!(matches!(phase, RequestPhase::Prefetching { .. }));
    assert!(effects
        .iter()
        .any(|e| matches!(e, SlotEffect::StartPrefetch { .. })));

    // 2) Poll-side timeout (~4s in logs) — cancel prefetch, drop staging → Preempted
    let (phase, _) = phase.apply(SlotEvent::PrefetchTimeout, &ctx);
    assert!(matches!(
        phase,
        RequestPhase::Preempted {
            recovered_from_failure: false
        }
    ));

    // 3) ~12ms later in logs: R2H finished — worker would emit TransferCompleted; in Preempted
    //    the reducer must ignore it (same as late PrefetchReady)
    let late_op = Uuid::new_v4();
    let (phase, _) = phase.apply(SlotEvent::TransferCompleted { operation_id: late_op }, &ctx);
    assert!(matches!(
        phase,
        RequestPhase::Preempted {
            recovered_from_failure: false
        }
    ));
    let (phase, _) = phase.apply(
        SlotEvent::PrefetchReady {
            blocks: vec![(), ()],
        },
        &ctx,
    );
    assert!(matches!(
        phase,
        RequestPhase::Preempted {
            recovered_from_failure: false
        }
    ));

    // 4) Fall back: scheduler-driven AcquireMatches → AwaitingLookup (fresh local lookup)
    let (phase, effects) = phase.apply(
        SlotEvent::AcquireMatches {
            num_computed_tokens: 0,
        },
        &ctx,
    );
    assert!(matches!(phase, RequestPhase::AwaitingLookup));
    assert!(
        effects
            .iter()
            .any(|e| matches!(e, SlotEffect::RunLocalLookup { num_computed_tokens: 0 })),
        "post-timeout recovery should re-run local lookup"
    );
}

/// v0.18 worker: G4/host prefetch completion is `TransferCompleted` with the same
/// `operation_id` as [`PrefetchState`] / [`SlotEffect::StartPrefetch`]. Reducer moves
/// to [`RequestPhase::OnboardReady`] (same end state as `PrefetchReady { blocks: [] }`).
#[test]
fn test_prefetching_transfer_completed_matching_operation_id_onboards() {
    let ctx = mock_ctx();
    let op_id = Uuid::new_v4();
    let prefetch = PrefetchState {
        operation_id: op_id,
        sequence_hashes: vec![0x1111],
        num_external_tokens: 16,
        started_at: std::time::Instant::now(),
    };
    let phase: TestPhase = RequestPhase::Prefetching {
        prefetch,
        host_staging: vec![()],
        disk_staging: vec![],
        prefetched_blocks_for_stats: 1,
    };
    let (new_phase, effects) =
        phase.apply(SlotEvent::TransferCompleted { operation_id: op_id }, &ctx);

    assert!(
        effects.is_empty(),
        "onboard-from-prefetch should not emit extra effects"
    );
    let RequestPhase::OnboardReady {
        num_external_tokens,
        host_staging,
        disk_staging,
        remote_hashes,
        disclosure,
        prefetched_blocks_for_stats,
    } = new_phase
    else {
        panic!(
            "expected OnboardReady after matching TransferCompleted; got {new_phase:?}"
        );
    };
    assert_eq!(num_external_tokens, 16);
    assert_eq!(host_staging.len(), 1);
    assert_eq!(disk_staging.len(), 0);
    assert_eq!(remote_hashes, vec![0x1111u64]);
    assert_eq!(disclosure, DisclosureState::Pending);
    assert_eq!(prefetched_blocks_for_stats, 1);
}

#[test]
fn test_prefetching_transfer_completed_mismatched_operation_id_is_noop() {
    let ctx = mock_ctx();
    let prefetch = PrefetchState {
        operation_id: Uuid::new_v4(),
        sequence_hashes: vec![0x1111],
        num_external_tokens: 16,
        started_at: std::time::Instant::now(),
    };
    let phase: TestPhase = RequestPhase::Prefetching {
        prefetch,
        host_staging: vec![()],
        disk_staging: vec![],
        prefetched_blocks_for_stats: 1,
    };
    let (new_phase, effects) = phase.apply(
        SlotEvent::TransferCompleted {
            operation_id: Uuid::new_v4(),
        },
        &ctx,
    );
    assert!(
        matches!(new_phase, RequestPhase::Prefetching { .. }),
        "unrelated transfer completion should not affect prefetch"
    );
    assert!(effects.is_empty());
}

#[test]
fn test_prefetching_failed_fallback_with_staging() {
    // (Prefetching, PrefetchFailed) + Fallback policy + non-empty staging
    // -> drop remote, keep host+disk -> OnboardReady
    let phase: TestPhase = RequestPhase::Prefetching {
        prefetch: mock_prefetch(),
        host_staging: vec![(), ()],
        disk_staging: vec![()],
        prefetched_blocks_for_stats: 2,
    };
    let (new_phase, _effects) = phase.apply(SlotEvent::PrefetchFailed, &mock_ctx());

    assert!(
        matches!(new_phase, RequestPhase::OnboardReady { .. }),
        "fallback with local staging -> OnboardReady"
    );
    if let RequestPhase::OnboardReady {
        host_staging,
        disk_staging,
        disclosure,
        ..
    } = &new_phase
    {
        assert_eq!(host_staging.len(), 2, "host staging preserved");
        assert_eq!(disk_staging.len(), 1, "disk staging preserved");
        assert_eq!(*disclosure, DisclosureState::Pending);
    }
}

#[test]
fn test_prefetching_failed_fallback_no_staging() {
    // (Prefetching, PrefetchFailed) + Fallback policy + no local staging
    // -> Preempted { recovered_from_failure: true }
    let phase: TestPhase = RequestPhase::Prefetching {
        prefetch: mock_prefetch(),
        host_staging: vec![],
        disk_staging: vec![],
        prefetched_blocks_for_stats: 0,
    };
    let (new_phase, _effects) = phase.apply(SlotEvent::PrefetchFailed, &mock_ctx());

    assert!(
        matches!(
            new_phase,
            RequestPhase::Preempted {
                recovered_from_failure: true
            }
        ),
        "no local staging to fall back to -> preempted"
    );
}

#[test]
fn test_prefetching_failed_abort_policy() {
    // (Prefetching, PrefetchFailed) + Abort policy -> always Preempted
    let phase: TestPhase = RequestPhase::Prefetching {
        prefetch: mock_prefetch(),
        host_staging: vec![(), ()],
        disk_staging: vec![()],
        prefetched_blocks_for_stats: 2,
    };
    let (new_phase, _effects) = phase.apply(SlotEvent::PrefetchFailed, &mock_ctx_abort());

    assert!(
        matches!(
            new_phase,
            RequestPhase::Preempted {
                recovered_from_failure: true
            }
        ),
        "abort policy ignores local staging"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Match reporting transitions
// Transition table rows:
//   (OnboardReady { Pending },  PollMatchReport) -> emit MatchReady -> Disclosed
//   (OnboardReady { Disclosed }, PollMatchReport) -> emit MatchDeferred -> Consumed
//   (OnboardReady { Consumed },  PollMatchReport) -> emit MatchDeferred -> Consumed
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_onboard_ready_pending_poll() {
    // First poll: Pending -> emit MatchReady { N } -> Disclosed
    let phase: TestPhase = RequestPhase::OnboardReady {
        num_external_tokens: 512,
        host_staging: vec![(), ()],
        disk_staging: vec![()],
        remote_hashes: vec![],
        disclosure: DisclosureState::Pending,
        prefetched_blocks_for_stats: 0,
    };
    let (new_phase, effects) = phase.apply(SlotEvent::PollMatchReport, &mock_ctx());

    assert!(matches!(
        new_phase,
        RequestPhase::OnboardReady {
            disclosure: DisclosureState::Disclosed,
            ..
        }
    ));
    assert!(effects
        .iter()
        .any(|e| matches!(e, SlotEffect::MatchReady { num_external_tokens: 512 })));
}

#[test]
fn test_onboard_ready_disclosed_poll() {
    // Second poll: Disclosed -> emit MatchDeferred -> Consumed
    let phase: TestPhase = RequestPhase::OnboardReady {
        num_external_tokens: 256,
        host_staging: vec![()],
        disk_staging: vec![],
        remote_hashes: vec![],
        disclosure: DisclosureState::Disclosed,
        prefetched_blocks_for_stats: 0,
    };
    let (new_phase, effects) = phase.apply(SlotEvent::PollMatchReport, &mock_ctx());

    assert!(matches!(
        new_phase,
        RequestPhase::OnboardReady {
            disclosure: DisclosureState::Consumed,
            ..
        }
    ));
    assert!(effects
        .iter()
        .any(|e| matches!(e, SlotEffect::MatchDeferred)));
}

#[test]
fn test_onboard_ready_consumed_poll() {
    // Third+ poll: Consumed -> emit MatchDeferred -> stays Consumed
    let phase: TestPhase = RequestPhase::OnboardReady {
        num_external_tokens: 128,
        host_staging: vec![],
        disk_staging: vec![],
        remote_hashes: vec![],
        disclosure: DisclosureState::Consumed,
        prefetched_blocks_for_stats: 0,
    };
    let (new_phase, effects) = phase.apply(SlotEvent::PollMatchReport, &mock_ctx());

    assert!(matches!(
        new_phase,
        RequestPhase::OnboardReady {
            disclosure: DisclosureState::Consumed,
            ..
        }
    ));
    assert!(effects
        .iter()
        .any(|e| matches!(e, SlotEffect::MatchDeferred)));
}

#[test]
fn test_full_disclosure_lifecycle() {
    // Walk through the entire Pending -> Disclosed -> Consumed -> Consumed cycle
    let phase: TestPhase = onboard_ready_pending(2, 1, 768);

    // Poll 1: Pending -> Disclosed
    let (phase, effects) = phase.apply(SlotEvent::PollMatchReport, &mock_ctx());
    assert!(matches!(
        phase,
        RequestPhase::OnboardReady {
            disclosure: DisclosureState::Disclosed,
            ..
        }
    ));
    assert!(effects
        .iter()
        .any(|e| matches!(e, SlotEffect::MatchReady { num_external_tokens: 768 })));

    // Poll 2: Disclosed -> Consumed
    let (phase, effects) = phase.apply(SlotEvent::PollMatchReport, &mock_ctx());
    assert!(matches!(
        phase,
        RequestPhase::OnboardReady {
            disclosure: DisclosureState::Consumed,
            ..
        }
    ));
    assert!(effects
        .iter()
        .any(|e| matches!(e, SlotEffect::MatchDeferred)));

    // Poll 3: Consumed -> Consumed (stable)
    let (phase, effects) = phase.apply(SlotEvent::PollMatchReport, &mock_ctx());
    assert!(matches!(
        phase,
        RequestPhase::OnboardReady {
            disclosure: DisclosureState::Consumed,
            ..
        }
    ));
    assert!(effects
        .iter()
        .any(|e| matches!(e, SlotEffect::MatchDeferred)));
}

// ═══════════════════════════════════════════════════════════════════════════
// Allocation and onboarding transitions
// Transition table rows:
//   (OnboardReady, AllocCompleted) -> emit EnqueueOnboardTransfer -> Onboarding
//   (Onboarding, ApplySchedulerOutput { tokens: [] }) -> Prefilling
//   (Onboarding, ApplySchedulerOutput { tokens: [..] }) -> Decoding
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_onboard_ready_alloc_completed() {
    // (OnboardReady, AllocCompleted) -> staging moved into EnqueueOnboardTransfer -> Onboarding
    let phase: TestPhase = RequestPhase::OnboardReady {
        num_external_tokens: 256,
        host_staging: vec![(), ()],
        disk_staging: vec![()],
        remote_hashes: vec![0xAAAA, 0xBBBB],
        disclosure: DisclosureState::Consumed,
        prefetched_blocks_for_stats: 2,
    };
    let (new_phase, effects) = phase.apply(
        SlotEvent::AllocCompleted {
            block_ids: vec![10, 11, 12],
            num_external_tokens: 256,
        },
        &mock_ctx(),
    );

    assert!(matches!(new_phase, RequestPhase::Onboarding { .. }));
    if let RequestPhase::Onboarding {
        num_external_tokens,
        ..
    } = &new_phase
    {
        assert_eq!(*num_external_tokens, 256);
    }

    assert!(effects
        .iter()
        .any(|e| matches!(e, SlotEffect::EnqueueOnboardTransfer { .. })));
    if let Some(SlotEffect::EnqueueOnboardTransfer {
        host_staging,
        disk_staging,
        remote_hashes,
        dst_blocks,
        num_external_tokens,
    }) = effects
        .iter()
        .find(|e| matches!(e, SlotEffect::EnqueueOnboardTransfer { .. }))
    {
        assert_eq!(host_staging.len(), 2);
        assert_eq!(disk_staging.len(), 1);
        assert_eq!(remote_hashes, &[0xAAAA, 0xBBBB]);
        assert_eq!(dst_blocks, &[10, 11, 12]);
        assert_eq!(*num_external_tokens, 256);
    }
}

#[test]
fn test_onboarding_scheduler_output_prefilling() {
    // (Onboarding, ApplySchedulerOutput { tokens: [] }) -> Prefilling
    let phase: TestPhase = RequestPhase::Onboarding {
        num_external_tokens: 256,
        ops: OperationTracker::new(),
    };
    let (new_phase, _effects) = phase.apply(make_scheduler_output_prefill(), &mock_ctx());

    assert!(matches!(new_phase, RequestPhase::Prefilling { .. }));
}

#[test]
fn test_onboarding_scheduler_output_decoding() {
    // (Onboarding, ApplySchedulerOutput { tokens: [..] }) -> Decoding
    let phase: TestPhase = RequestPhase::Onboarding {
        num_external_tokens: 256,
        ops: OperationTracker::new(),
    };
    let (new_phase, _effects) = phase.apply(make_scheduler_output_decode(), &mock_ctx());

    assert!(matches!(new_phase, RequestPhase::Decoding { .. }));
    if let RequestPhase::Decoding {
        iteration_first_scheduled,
        ..
    } = &new_phase
    {
        assert!(*iteration_first_scheduled > 0 || *iteration_first_scheduled == 0);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Forward pass: Decoding transitions
// Transition table rows:
//   (Decoding, ApplySchedulerOutput) -> stays Decoding (offloads are position-based in slot_runtime, not reducer effects)
//   (Decoding, RequestFinished) + ops -> Finishing
//   (Decoding, RequestFinished) + no ops -> emit RecordCacheStats -> Finished
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_decoding_scheduler_output() {
    // (Decoding, ApplySchedulerOutput) -> stays Decoding
    let phase: TestPhase = RequestPhase::Decoding {
        ops: OperationTracker::new(),
        iteration_first_scheduled: 1,
    };
    let (new_phase, _effects) = phase.apply(make_scheduler_output_decode(), &mock_ctx());

    assert!(matches!(new_phase, RequestPhase::Decoding { .. }));
    if let RequestPhase::Decoding {
        iteration_first_scheduled,
        ..
    } = &new_phase
    {
        assert_eq!(*iteration_first_scheduled, 1, "preserves first-scheduled");
    }
}

#[test]
fn test_decoding_request_finished_with_ops() {
    // (Decoding, RequestFinished) + ops.has_any() -> Finishing { ops }
    let phase: TestPhase = RequestPhase::Decoding {
        ops: tracker_with_dispatched("finish-test", 2),
        iteration_first_scheduled: 5,
    };
    let (new_phase, _effects) = phase.apply(SlotEvent::RequestFinished, &mock_ctx());

    assert!(matches!(new_phase, RequestPhase::Finishing { .. }));
    if let RequestPhase::Finishing { ops } = &new_phase {
        assert!(ops.has_any());
    }
}

#[test]
fn test_decoding_request_finished_no_ops() {
    // (Decoding, RequestFinished) + no ops -> emit RecordCacheStats -> Finished
    let phase: TestPhase = RequestPhase::Decoding {
        ops: OperationTracker::new(),
        iteration_first_scheduled: 5,
    };
    let (new_phase, effects) = phase.apply(SlotEvent::RequestFinished, &mock_ctx());

    assert!(matches!(new_phase, RequestPhase::Finished));
    assert!(effects
        .iter()
        .any(|e| matches!(e, SlotEffect::RecordCacheStats { .. })));
}

// ═══════════════════════════════════════════════════════════════════════════
// Skipped-step transitions
// Transition table rows:
//   (Prefilling, MarkSkipped) -> SkippedPrefill
//   (SkippedPrefill, ApplySchedulerOutput { tokens: [] }) -> Prefilling
//   (SkippedPrefill, ApplySchedulerOutput { tokens: [..] }) -> Decoding
//   (Decoding, MarkSkipped) -> SkippedDecode
//   (SkippedDecode, ApplySchedulerOutput) -> Decoding
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_prefilling_mark_skipped() {
    // (Prefilling, MarkSkipped) -> SkippedPrefill
    let phase: TestPhase = RequestPhase::Prefilling {
        iteration_first_scheduled: 10,
    };
    let (new_phase, effects) = phase.apply(SlotEvent::MarkSkipped, &mock_ctx());

    assert!(matches!(new_phase, RequestPhase::SkippedPrefill));
    assert!(effects.is_empty());
}

#[test]
fn test_skipped_prefill_scheduler_output_empty_tokens() {
    // (SkippedPrefill, ApplySchedulerOutput { tokens: [] }) -> Prefilling
    let phase: TestPhase = RequestPhase::SkippedPrefill;
    let (new_phase, _effects) = phase.apply(make_scheduler_output_prefill(), &mock_ctx());

    assert!(matches!(new_phase, RequestPhase::Prefilling { .. }));
}

#[test]
fn test_skipped_prefill_scheduler_output_with_tokens() {
    // (SkippedPrefill, ApplySchedulerOutput { tokens: [..] }) -> Decoding
    let phase: TestPhase = RequestPhase::SkippedPrefill;
    let (new_phase, _effects) = phase.apply(make_scheduler_output_decode(), &mock_ctx());

    assert!(matches!(new_phase, RequestPhase::Decoding { .. }));
}

#[test]
fn test_decoding_mark_skipped() {
    // (Decoding, MarkSkipped) -> SkippedDecode
    let phase: TestPhase = RequestPhase::Decoding {
        ops: OperationTracker::new(),
        iteration_first_scheduled: 15,
    };
    let (new_phase, effects) = phase.apply(SlotEvent::MarkSkipped, &mock_ctx());

    assert!(matches!(new_phase, RequestPhase::SkippedDecode { .. }));
    assert!(effects.is_empty());
}

#[test]
fn test_skipped_decode_scheduler_output() {
    // (SkippedDecode, ApplySchedulerOutput) -> Decoding
    let phase: TestPhase = RequestPhase::SkippedDecode { ops: OperationTracker::new() };
    let (new_phase, _effects) = phase.apply(make_scheduler_output_decode(), &mock_ctx());

    assert!(matches!(new_phase, RequestPhase::Decoding { .. }));
}

// ═══════════════════════════════════════════════════════════════════════════
// Finishing transitions
// Transition table rows:
//   (Finishing, TransferCompleted) + ops drained -> emit RecordCacheStats -> Finished
//   (Finishing, TransferCompleted) + ops remain  -> Finishing
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_finishing_transfer_completed_drained() {
    // (Finishing, TransferCompleted) + all ops done -> emit RecordCacheStats -> Finished
    let op_id = Uuid::new_v4();
    let mut tracker = OperationTracker::new();
    tracker.append_pending(mock_load_request("finish-drain", vec![1]));
    let _ = tracker.take_pending_for_dispatch();

    let phase: TestPhase = RequestPhase::Finishing { ops: tracker };
    let (new_phase, effects) = phase.apply(
        SlotEvent::TransferCompleted {
            operation_id: op_id,
        },
        &mock_ctx(),
    );

    assert!(
        matches!(new_phase, RequestPhase::Finished),
        "single dispatched op completed -> Finished"
    );
    assert!(effects
        .iter()
        .any(|e| matches!(e, SlotEffect::RecordCacheStats { .. })));
}

#[test]
fn test_finishing_transfer_completed_not_drained() {
    // (Finishing, TransferCompleted) + ops remain -> stays Finishing
    let mut tracker = OperationTracker::new();
    tracker.append_pending(mock_load_request("finish-partial", vec![1]));
    tracker.append_pending(mock_load_request("finish-partial", vec![2]));
    let _ = tracker.take_pending_for_dispatch();

    let phase: TestPhase = RequestPhase::Finishing { ops: tracker };
    let (new_phase, effects) = phase.apply(
        SlotEvent::TransferCompleted {
            operation_id: Uuid::new_v4(),
        },
        &mock_ctx(),
    );

    assert!(
        matches!(new_phase, RequestPhase::Finishing { .. }),
        "one of two ops completed -> still Finishing"
    );
    assert!(!effects
        .iter()
        .any(|e| matches!(e, SlotEffect::RecordCacheStats { .. })));
}

// ═══════════════════════════════════════════════════════════════════════════
// Preemption and retry
// Transition table rows:
//   (Preempted, AcquireMatches) -> fresh lookup (retry)
//   (AnyPhaseWithStaging, Preempt) -> staging dropped -> Preempted
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_preempted_acquire_matches() {
    // (Preempted, AcquireMatches) -> emit RunLocalLookup -> AwaitingLookup
    let phase: TestPhase = RequestPhase::Preempted {
        recovered_from_failure: false,
    };
    let (new_phase, effects) = phase.apply(
        SlotEvent::AcquireMatches {
            num_computed_tokens: 64,
        },
        &mock_ctx(),
    );

    assert!(matches!(new_phase, RequestPhase::AwaitingLookup));
    assert!(effects
        .iter()
        .any(|e| matches!(e, SlotEffect::RunLocalLookup { num_computed_tokens: 64 })));
}

#[test]
fn test_preempted_recovered_acquire_matches() {
    // Retry after failure recovery should also work
    let phase: TestPhase = RequestPhase::Preempted {
        recovered_from_failure: true,
    };
    let (new_phase, effects) = phase.apply(
        SlotEvent::AcquireMatches {
            num_computed_tokens: 128,
        },
        &mock_ctx(),
    );

    assert!(matches!(new_phase, RequestPhase::AwaitingLookup));
    assert!(effects
        .iter()
        .any(|e| matches!(e, SlotEffect::RunLocalLookup { .. })));
}

#[test]
fn test_preempt_during_prefetching() {
    // (Prefetching, Preempt) -> staging dropped -> Preempted
    let phase: TestPhase = RequestPhase::Prefetching {
        prefetch: mock_prefetch(),
        host_staging: vec![(), ()],
        disk_staging: vec![()],
        prefetched_blocks_for_stats: 2,
    };
    let (new_phase, _effects) = phase.apply(SlotEvent::Preempt, &mock_ctx());

    assert!(matches!(new_phase, RequestPhase::Preempted { .. }));
}

#[test]
fn test_preempt_during_onboard_ready() {
    // (OnboardReady, Preempt) -> staging dropped -> Preempted
    let phase: TestPhase = onboard_ready_pending(3, 1, 512);
    let (new_phase, _effects) = phase.apply(SlotEvent::Preempt, &mock_ctx());

    assert!(matches!(new_phase, RequestPhase::Preempted { .. }));
}

#[test]
fn test_preempt_during_decoding() {
    // (Decoding, Preempt) -> Preempted
    let phase: TestPhase = RequestPhase::Decoding {
        ops: tracker_with_dispatched("preempt-decode", 1),
        iteration_first_scheduled: 3,
    };
    let (new_phase, _effects) = phase.apply(SlotEvent::Preempt, &mock_ctx());

    assert!(matches!(new_phase, RequestPhase::Preempted { .. }));
}

#[test]
fn test_preempt_during_onboarding() {
    // (Onboarding, Preempt) -> Preempted
    let phase: TestPhase = RequestPhase::Onboarding {
        num_external_tokens: 256,
        ops: OperationTracker::new(),
    };
    let (new_phase, _effects) = phase.apply(SlotEvent::Preempt, &mock_ctx());

    assert!(matches!(new_phase, RequestPhase::Preempted { .. }));
}

#[test]
fn test_preempt_during_prefilling() {
    // (Prefilling, Preempt) -> Preempted
    let phase: TestPhase = RequestPhase::Prefilling {
        iteration_first_scheduled: 7,
    };
    let (new_phase, _effects) = phase.apply(SlotEvent::Preempt, &mock_ctx());

    assert!(matches!(new_phase, RequestPhase::Preempted { .. }));
}

#[test]
fn test_preempt_during_looking_up() {
    // (LookingUp, Preempt) -> staging dropped -> Preempted
    let phase: TestPhase = RequestPhase::LookingUp {
        host_blocks: vec![(), ()],
        disk_blocks: vec![()],
    };
    let (new_phase, _effects) = phase.apply(SlotEvent::Preempt, &mock_ctx());

    assert!(matches!(new_phase, RequestPhase::Preempted { .. }));
}

// ═══════════════════════════════════════════════════════════════════════════
// Reset transitions
// Transition table row:
//   (AnyPhase, Reset) -> everything consumed -> Initialized
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_reset_from_any_phase() {
    let phases: Vec<TestPhase> = vec![
        RequestPhase::Initialized,
        RequestPhase::AwaitingLookup,
        RequestPhase::LookingUp {
            host_blocks: vec![()],
            disk_blocks: vec![],
        },
        RequestPhase::Prefetching {
            prefetch: mock_prefetch(),
            host_staging: vec![()],
            disk_staging: vec![],
            prefetched_blocks_for_stats: 0,
        },
        RequestPhase::OnboardReady {
            num_external_tokens: 256,
            host_staging: vec![()],
            disk_staging: vec![],
            remote_hashes: vec![],
            disclosure: DisclosureState::Pending,
            prefetched_blocks_for_stats: 0,
        },
        RequestPhase::Onboarding {
            num_external_tokens: 256,
            ops: OperationTracker::new(),
        },
        RequestPhase::Prefilling {
            iteration_first_scheduled: 1,
        },
        RequestPhase::SkippedPrefill,
        RequestPhase::Decoding {
            ops: OperationTracker::new(),
            iteration_first_scheduled: 1,
        },
        RequestPhase::SkippedDecode { ops: OperationTracker::new() },
        RequestPhase::Finishing {
            ops: OperationTracker::new(),
        },
        RequestPhase::Finished,
        RequestPhase::Preempted {
            recovered_from_failure: false,
        },
        RequestPhase::Preempted {
            recovered_from_failure: true,
        },
    ];

    for (i, phase) in phases.into_iter().enumerate() {
        let (new_phase, _effects) = phase.apply(SlotEvent::Reset, &mock_ctx());
        assert!(
            matches!(new_phase, RequestPhase::Initialized),
            "phase index {} did not reset to Initialized",
            i
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Late event no-ops
// Events that arrive after the relevant phase has passed are silent no-ops.
// The exhaustive match in apply() handles them explicitly.
// ═══════════════════════════════════════════════════════════════════════════

#[rstest]
#[case::prefetch_ready_in_initialized(
    RequestPhase::Initialized,
    SlotEvent::PrefetchReady { blocks: vec![] }
)]
#[case::transfer_completed_in_initialized(
    RequestPhase::Initialized,
    SlotEvent::TransferCompleted { operation_id: Uuid::nil() }
)]
#[case::poll_in_finished(
    RequestPhase::Finished,
    SlotEvent::PollMatchReport
)]
#[case::prefetch_timeout_in_finished(
    RequestPhase::Finished,
    SlotEvent::PrefetchTimeout
)]
#[case::prefetch_ready_in_preempted(
    RequestPhase::Preempted { recovered_from_failure: false },
    SlotEvent::PrefetchReady { blocks: vec![] }
)]
#[case::transfer_completed_in_preempted(
    RequestPhase::Preempted { recovered_from_failure: true },
    SlotEvent::TransferCompleted { operation_id: Uuid::nil() }
)]
#[case::prefetch_failed_in_finished(
    RequestPhase::Finished,
    SlotEvent::PrefetchFailed
)]
#[case::remote_lookup_closed_in_finished(
    RequestPhase::Finished,
    SlotEvent::RemoteLookupClosed
)]
#[case::mark_skipped_in_initialized(
    RequestPhase::Initialized,
    SlotEvent::MarkSkipped
)]
#[case::alloc_completed_in_finished(
    RequestPhase::Finished,
    SlotEvent::AllocCompleted { block_ids: vec![], num_external_tokens: 0 }
)]
#[case::request_finished_in_preempted(
    RequestPhase::Preempted { recovered_from_failure: false },
    SlotEvent::RequestFinished
)]
fn test_late_events_are_noop(#[case] phase: TestPhase, #[case] event: TestEvent) {
    let (_new_phase, effects) = phase.apply(event, &mock_ctx());
    assert!(
        effects.is_empty()
            || effects
                .iter()
                .all(|e| matches!(e, SlotEffect::Diag { .. })),
        "late events should produce no effects (or diagnostics only)"
    );
}

// More targeted late-event checks
#[test]
fn test_late_prefetch_ready_after_preempt_preserves_flag() {
    let phase: TestPhase = RequestPhase::Preempted {
        recovered_from_failure: true,
    };
    let (new_phase, effects) = phase.apply(
        SlotEvent::PrefetchReady {
            blocks: vec![(), ()],
        },
        &mock_ctx(),
    );
    assert!(matches!(
        new_phase,
        RequestPhase::Preempted {
            recovered_from_failure: true
        }
    ));
    assert!(effects.is_empty() || effects.iter().all(|e| matches!(e, SlotEffect::Diag { .. })));
}

#[test]
fn test_late_transfer_completed_in_preempted() {
    let phase: TestPhase = RequestPhase::Preempted {
        recovered_from_failure: false,
    };
    let (new_phase, effects) = phase.apply(
        SlotEvent::TransferCompleted {
            operation_id: Uuid::new_v4(),
        },
        &mock_ctx(),
    );
    assert!(matches!(
        new_phase,
        RequestPhase::Preempted {
            recovered_from_failure: false
        }
    ));
    assert!(effects.is_empty() || effects.iter().all(|e| matches!(e, SlotEffect::Diag { .. })));
}

#[test]
fn test_late_poll_match_report_in_decoding() {
    // PollMatchReport only makes sense in OnboardReady; should be no-op elsewhere
    let phase: TestPhase = RequestPhase::Decoding {
        ops: OperationTracker::new(),
        iteration_first_scheduled: 5,
    };
    let (new_phase, effects) = phase.apply(SlotEvent::PollMatchReport, &mock_ctx());
    assert!(matches!(new_phase, RequestPhase::Decoding { .. }));
    assert!(effects.is_empty() || effects.iter().all(|e| matches!(e, SlotEffect::Diag { .. })));
}

// ═══════════════════════════════════════════════════════════════════════════
// SlotState compatibility views
// The reducer exposes as_slot_state() for backward-compat with existing
// vLLM scheduler methods that inspect SlotState.
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_slot_state_compatibility_view() {
    // Verify every phase variant maps to the correct SlotState.
    let cases: Vec<(TestPhase, SlotState)> = vec![
        (RequestPhase::Initialized, SlotState::Initialized),
        (
            RequestPhase::OnboardReady {
                num_external_tokens: 512,
                host_staging: vec![],
                disk_staging: vec![],
                remote_hashes: vec![],
                disclosure: DisclosureState::Pending,
                prefetched_blocks_for_stats: 0,
            },
            SlotState::OnboardStaged(512),
        ),
        (
            RequestPhase::Onboarding {
                num_external_tokens: 256,
                ops: OperationTracker::new(),
            },
            SlotState::Onboarding(256),
        ),
        (
            RequestPhase::Prefilling {
                iteration_first_scheduled: 1,
            },
            SlotState::Prefilling,
        ),
        (RequestPhase::SkippedPrefill, SlotState::SkippedPrefill),
        (
            RequestPhase::Decoding {
                ops: OperationTracker::new(),
                iteration_first_scheduled: 1,
            },
            SlotState::Decoding,
        ),
        (RequestPhase::SkippedDecode { ops: OperationTracker::new() }, SlotState::SkippedDecode),
        (
            RequestPhase::Finishing {
                ops: OperationTracker::new(),
            },
            SlotState::Finishing,
        ),
        (RequestPhase::Finished, SlotState::Finished),
        (
            RequestPhase::Preempted {
                recovered_from_failure: false,
            },
            SlotState::Preempted,
        ),
    ];

    for (i, (phase, expected_state)) in cases.into_iter().enumerate() {
        let actual = phase.as_slot_state();
        assert_eq!(
            actual, expected_state,
            "case {}: {:?} != {:?}",
            i, actual, expected_state
        );
    }
}

#[test]
fn test_slot_state_onboard_staged_carries_token_count() {
    for n in [0_usize, 1, 128, 120064, usize::MAX] {
        let phase: TestPhase = RequestPhase::OnboardReady {
            num_external_tokens: n,
            host_staging: vec![],
            disk_staging: vec![],
            remote_hashes: vec![],
            disclosure: DisclosureState::Pending,
            prefetched_blocks_for_stats: 0,
        };
        assert_eq!(phase.as_slot_state(), SlotState::OnboardStaged(n));
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Edge cases and compound scenarios
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_initialized_is_pure_no_effects_on_construction() {
    // Verifying that the initial phase produces the expected effect on first event
    let phase: TestPhase = RequestPhase::Initialized;
    let (new_phase, effects) = phase.apply(
        SlotEvent::AcquireMatches {
            num_computed_tokens: 42,
        },
        &mock_ctx(),
    );
    assert!(matches!(new_phase, RequestPhase::AwaitingLookup));
    assert_eq!(effects.len(), 1);
    if let SlotEffect::RunLocalLookup {
        num_computed_tokens,
    } = &effects[0]
    {
        assert_eq!(*num_computed_tokens, 42);
    } else {
        panic!("expected RunLocalLookup effect");
    }
}

#[test]
fn test_onboard_ready_zero_external_tokens() {
    // OnboardReady can have zero tokens (degenerate case, e.g. all computed)
    let phase: TestPhase = RequestPhase::OnboardReady {
        num_external_tokens: 0,
        host_staging: vec![],
        disk_staging: vec![],
        remote_hashes: vec![],
        disclosure: DisclosureState::Pending,
        prefetched_blocks_for_stats: 0,
    };
    let (new_phase, effects) = phase.apply(SlotEvent::PollMatchReport, &mock_ctx());

    assert!(matches!(
        new_phase,
        RequestPhase::OnboardReady {
            disclosure: DisclosureState::Disclosed,
            ..
        }
    ));
    assert!(effects
        .iter()
        .any(|e| matches!(e, SlotEffect::MatchReady { num_external_tokens: 0 })));
}

#[test]
fn test_preempt_from_awaiting_lookup() {
    // AwaitingLookup has no staging but should still be preemptable
    let phase: TestPhase = RequestPhase::AwaitingLookup;
    let (new_phase, _effects) = phase.apply(SlotEvent::Preempt, &mock_ctx());

    assert!(matches!(new_phase, RequestPhase::Preempted { .. }));
}

#[test]
fn test_full_happy_path_local_only() {
    // Walk through: Initialized -> AwaitingLookup -> OnboardReady -> Onboarding
    //            -> Prefilling -> Decoding -> Finished
    let ctx = mock_ctx();

    // Step 1: Initialized -> AcquireMatches -> AwaitingLookup
    let phase: TestPhase = RequestPhase::Initialized;
    let (phase, effects) = phase.apply(
        SlotEvent::AcquireMatches {
            num_computed_tokens: 0,
        },
        &ctx,
    );
    assert!(matches!(phase, RequestPhase::AwaitingLookup));
    assert!(effects
        .iter()
        .any(|e| matches!(e, SlotEffect::RunLocalLookup { .. })));

    // Step 2: AwaitingLookup -> LocalLookupCompleted (local only) -> OnboardReady
    let (phase, _effects) = phase.apply(
        SlotEvent::LocalLookupCompleted {
            host_blocks: vec![(), (), ()],
            disk_blocks: vec![()],
            remote_candidates: vec![],
        },
        &ctx,
    );
    assert!(matches!(
        phase,
        RequestPhase::OnboardReady {
            disclosure: DisclosureState::Pending,
            ..
        }
    ));

    // Step 3: PollMatchReport -> Disclosed
    let (phase, effects) = phase.apply(SlotEvent::PollMatchReport, &ctx);
    assert!(matches!(
        phase,
        RequestPhase::OnboardReady {
            disclosure: DisclosureState::Disclosed,
            ..
        }
    ));
    assert!(effects
        .iter()
        .any(|e| matches!(e, SlotEffect::MatchReady { .. })));

    // Step 4: AllocCompleted -> Onboarding
    let (phase, effects) = phase.apply(
        SlotEvent::AllocCompleted {
            block_ids: vec![10, 11, 12, 13],
            num_external_tokens: 64,
        },
        &ctx,
    );
    assert!(matches!(phase, RequestPhase::Onboarding { .. }));
    assert!(effects
        .iter()
        .any(|e| matches!(e, SlotEffect::EnqueueOnboardTransfer { .. })));

    // Step 5: ApplySchedulerOutput (empty tokens) -> Prefilling
    let (phase, _effects) = phase.apply(make_scheduler_output_prefill(), &ctx);
    assert!(matches!(phase, RequestPhase::Prefilling { .. }));

    // Step 6: ApplySchedulerOutput (with tokens) -> Decoding
    let (phase, _effects) = phase.apply(make_scheduler_output_decode(), &ctx);
    assert!(matches!(phase, RequestPhase::Decoding { .. }));

    // Step 7: RequestFinished (no ops) -> Finished
    let (phase, effects) = phase.apply(SlotEvent::RequestFinished, &ctx);
    assert!(matches!(phase, RequestPhase::Finished));
    assert!(effects
        .iter()
        .any(|e| matches!(e, SlotEffect::RecordCacheStats { .. })));
}

#[test]
fn test_full_happy_path_with_remote_prefetch() {
    // Walk through the remote path: Initialized -> AwaitingLookup -> LookingUp
    //   -> Prefetching -> OnboardReady -> Onboarding -> Decoding -> Finished
    let ctx = mock_ctx();

    // Step 1: Initialized -> AwaitingLookup
    let phase: TestPhase = RequestPhase::Initialized;
    let (phase, _) = phase.apply(
        SlotEvent::AcquireMatches {
            num_computed_tokens: 0,
        },
        &ctx,
    );

    // Step 2: AwaitingLookup -> LookingUp (has remote candidates)
    let (phase, effects) = phase.apply(
        SlotEvent::LocalLookupCompleted {
            host_blocks: vec![(), ()],
            disk_blocks: vec![],
            remote_candidates: vec![0x1111, 0x2222],
        },
        &ctx,
    );
    assert!(matches!(phase, RequestPhase::LookingUp { .. }));
    assert!(effects
        .iter()
        .any(|e| matches!(e, SlotEffect::StartRemoteLookup { .. })));

    // Step 3: LookingUp -> Prefetching (remote found hashes)
    let (phase, effects) = phase.apply(
        SlotEvent::RemoteLookupCompleted {
            matches: vec![mock_remote_match()],
        },
        &ctx,
    );
    assert!(matches!(phase, RequestPhase::Prefetching { .. }));
    assert!(effects
        .iter()
        .any(|e| matches!(e, SlotEffect::StartPrefetch { .. })));

    // Step 4: Prefetching -> OnboardReady (prefetch succeeded)
    let (phase, _) = phase.apply(
        SlotEvent::PrefetchReady {
            blocks: vec![(), ()],
        },
        &ctx,
    );
    assert!(matches!(
        phase,
        RequestPhase::OnboardReady {
            disclosure: DisclosureState::Pending,
            ..
        }
    ));

    // Step 5: Poll -> Disclosed, Alloc -> Onboarding
    let (phase, _) = phase.apply(SlotEvent::PollMatchReport, &ctx);
    let (phase, _) = phase.apply(
        SlotEvent::AllocCompleted {
            block_ids: vec![20, 21],
            num_external_tokens: 128,
        },
        &ctx,
    );
    assert!(matches!(phase, RequestPhase::Onboarding { .. }));

    // Step 6: Schedule -> Decoding (with tokens)
    let (phase, _) = phase.apply(make_scheduler_output_decode(), &ctx);
    assert!(matches!(phase, RequestPhase::Decoding { .. }));

    // Step 7: Finish
    let (phase, _) = phase.apply(SlotEvent::RequestFinished, &ctx);
    assert!(matches!(phase, RequestPhase::Finished));
}

#[test]
fn test_preempt_and_retry_cycle() {
    // Preempted slot retries via AcquireMatches and completes
    let ctx = mock_ctx();

    // Start -> AwaitingLookup
    let phase: TestPhase = RequestPhase::Initialized;
    let (phase, _) = phase.apply(
        SlotEvent::AcquireMatches {
            num_computed_tokens: 0,
        },
        &ctx,
    );

    // -> OnboardReady
    let (phase, _) = phase.apply(
        SlotEvent::LocalLookupCompleted {
            host_blocks: vec![()],
            disk_blocks: vec![],
            remote_candidates: vec![],
        },
        &ctx,
    );

    // Preempt
    let (phase, _) = phase.apply(SlotEvent::Preempt, &ctx);
    assert!(matches!(phase, RequestPhase::Preempted { .. }));

    // Retry
    let (phase, effects) = phase.apply(
        SlotEvent::AcquireMatches {
            num_computed_tokens: 32,
        },
        &ctx,
    );
    assert!(matches!(phase, RequestPhase::AwaitingLookup));
    assert!(effects
        .iter()
        .any(|e| matches!(e, SlotEffect::RunLocalLookup { num_computed_tokens: 32 })));
}

#[test]
fn test_skipped_prefill_then_decode_preserves_iteration() {
    // SkippedPrefill -> Prefilling -> MarkSkipped -> SkippedPrefill -> Decoding
    let ctx = mock_ctx();

    let phase: TestPhase = RequestPhase::SkippedPrefill;

    // Re-schedule as prefill
    let (phase, _) = phase.apply(make_scheduler_output_prefill(), &ctx);
    assert!(matches!(phase, RequestPhase::Prefilling { .. }));

    // Skip again
    let (phase, _) = phase.apply(SlotEvent::MarkSkipped, &ctx);
    assert!(matches!(phase, RequestPhase::SkippedPrefill));

    // Now promote to decode
    let (phase, _) = phase.apply(make_scheduler_output_decode(), &ctx);
    assert!(matches!(phase, RequestPhase::Decoding { .. }));
}

#[test]
fn test_skipped_decode_round_trip() {
    // Decoding -> MarkSkipped -> SkippedDecode -> ApplySchedulerOutput -> Decoding
    let ctx = mock_ctx();

    let phase: TestPhase = RequestPhase::Decoding {
        ops: OperationTracker::new(),
        iteration_first_scheduled: 10,
    };

    let (phase, _) = phase.apply(SlotEvent::MarkSkipped, &ctx);
    assert!(matches!(phase, RequestPhase::SkippedDecode { .. }));

    let (phase, _) = phase.apply(make_scheduler_output_decode(), &ctx);
    assert!(matches!(phase, RequestPhase::Decoding { .. }));
}

#[test]
fn test_finishing_with_multiple_completions() {
    // Two dispatched ops: complete both sequentially -> Finished
    let mut tracker = OperationTracker::new();
    tracker.append_pending(mock_load_request("multi-fin", vec![1]));
    tracker.append_pending(mock_load_request("multi-fin", vec![2]));
    let _ = tracker.take_pending_for_dispatch();

    let phase: TestPhase = RequestPhase::Finishing { ops: tracker };

    // Complete first op
    let (phase, effects) = phase.apply(
        SlotEvent::TransferCompleted {
            operation_id: Uuid::new_v4(),
        },
        &mock_ctx(),
    );
    assert!(
        matches!(phase, RequestPhase::Finishing { .. }),
        "one of two -> still finishing"
    );
    assert!(!effects
        .iter()
        .any(|e| matches!(e, SlotEffect::RecordCacheStats { .. })));

    // Complete second op
    let (phase, effects) = phase.apply(
        SlotEvent::TransferCompleted {
            operation_id: Uuid::new_v4(),
        },
        &mock_ctx(),
    );
    assert!(
        matches!(phase, RequestPhase::Finished),
        "both completed -> Finished"
    );
    assert!(effects
        .iter()
        .any(|e| matches!(e, SlotEffect::RecordCacheStats { .. })));
}

// ═══════════════════════════════════════════════════════════════════════════
// Part B: Exhaustive transition matrix
//
// Enumerates every (phase × event) pair and classifies each as a real
// transition, a data-dependent transition, or a no-op. A single test
// iterates the full 13×17 = 221 matrix and asserts:
//   - NoOp pairs stay in the same phase with empty/Diag-only effects
//   - Every pair has a classification (no silent gaps)
// ═══════════════════════════════════════════════════════════════════════════

fn make_phase(name: &str) -> TestPhase {
    match name {
        "Initialized" => RequestPhase::Initialized,
        "AwaitingLookup" => RequestPhase::AwaitingLookup,
        "LookingUp" => RequestPhase::LookingUp {
            host_blocks: vec![()],
            disk_blocks: vec![()],
        },
        "Prefetching" => RequestPhase::Prefetching {
            prefetch: mock_prefetch(),
            host_staging: vec![()],
            disk_staging: vec![],
            prefetched_blocks_for_stats: 1,
        },
        "OnboardReady" => RequestPhase::OnboardReady {
            num_external_tokens: 16,
            host_staging: vec![()],
            disk_staging: vec![],
            remote_hashes: vec![0x1111],
            disclosure: DisclosureState::Pending,
            prefetched_blocks_for_stats: 0,
        },
        "Onboarding" => RequestPhase::Onboarding {
            num_external_tokens: 16,
            ops: OperationTracker::new(),
        },
        "Prefilling" => RequestPhase::Prefilling {
            iteration_first_scheduled: 1,
        },
        "SkippedPrefill" => RequestPhase::SkippedPrefill,
        "Decoding" => RequestPhase::Decoding {
            ops: OperationTracker::new(),
            iteration_first_scheduled: 1,
        },
        "SkippedDecode" => RequestPhase::SkippedDecode {
            ops: OperationTracker::new(),
        },
        "Finishing" => RequestPhase::Finishing {
            ops: {
                let mut t = OperationTracker::new();
                t.append_pending(mock_load_request("matrix", vec![1]));
                let _ = t.take_pending_for_dispatch();
                t
            },
        },
        "Finished" => RequestPhase::Finished,
        "Preempted" => RequestPhase::Preempted {
            recovered_from_failure: false,
        },
        _ => panic!("unknown phase: {name}"),
    }
}

fn make_event(name: &str) -> TestEvent {
    match name {
        "AcquireMatches" => SlotEvent::AcquireMatches { num_computed_tokens: 0 },
        "LocalLookupCompleted" => SlotEvent::LocalLookupCompleted {
            host_blocks: vec![()],
            disk_blocks: vec![],
            remote_candidates: vec![],
        },
        "RemoteLookupCompleted" => SlotEvent::RemoteLookupCompleted {
            matches: vec![(0xAB, 0xCD)],
        },
        "RemoteLookupClosed" => SlotEvent::RemoteLookupClosed,
        "PrefetchReady" => SlotEvent::PrefetchReady { blocks: vec![()] },
        "PrefetchTimeout" => SlotEvent::PrefetchTimeout,
        "PrefetchFailed" => SlotEvent::PrefetchFailed,
        "PollMatchReport" => SlotEvent::PollMatchReport,
        "AllocCompleted" => SlotEvent::AllocCompleted {
            block_ids: vec![10, 11],
            num_external_tokens: 32,
        },
        "ApplySchedulerOutput" => SlotEvent::ApplySchedulerOutput {
            tokens: vec![42],
            block_ids: vec![1, 2],
            num_computed_tokens: 16,
            num_scheduled_tokens: 1,
            priorities: None,
            iteration: 1,
        },
        "MarkSkipped" => SlotEvent::MarkSkipped,
        "TransferDispatched" => SlotEvent::TransferDispatched { count: 1 },
        "TransferCompleted" => SlotEvent::TransferCompleted {
            operation_id: Uuid::nil(),
        },
        "TransferFailed" => SlotEvent::TransferFailed {
            operation_id: Uuid::nil(),
        },
        "RequestFinished" => SlotEvent::RequestFinished,
        "Preempt" => SlotEvent::Preempt,
        "Reset" => SlotEvent::Reset,
        _ => panic!("unknown event: {name}"),
    }
}

const ALL_PHASE_NAMES: &[&str] = &[
    "Initialized", "AwaitingLookup", "LookingUp", "Prefetching",
    "OnboardReady", "Onboarding", "Prefilling", "SkippedPrefill",
    "Decoding", "SkippedDecode", "Finishing", "Finished", "Preempted",
];

const ALL_EVENT_NAMES: &[&str] = &[
    "AcquireMatches", "LocalLookupCompleted", "RemoteLookupCompleted",
    "RemoteLookupClosed", "PrefetchReady", "PrefetchTimeout", "PrefetchFailed",
    "PollMatchReport", "AllocCompleted", "ApplySchedulerOutput", "MarkSkipped",
    "TransferDispatched", "TransferCompleted", "TransferFailed", "RequestFinished",
    "Preempt", "Reset",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Expectation {
    Transitions,
    NoOp,
    DataDependent,
}

fn transition_table() -> std::collections::HashMap<(&'static str, &'static str), Expectation> {
    use Expectation::*;
    let entries: Vec<((&str, &str), Expectation)> = vec![
        // Reset from any phase
        (("Initialized", "Reset"), Transitions),
        (("AwaitingLookup", "Reset"), Transitions),
        (("LookingUp", "Reset"), Transitions),
        (("Prefetching", "Reset"), Transitions),
        (("OnboardReady", "Reset"), Transitions),
        (("Onboarding", "Reset"), Transitions),
        (("Prefilling", "Reset"), Transitions),
        (("SkippedPrefill", "Reset"), Transitions),
        (("Decoding", "Reset"), Transitions),
        (("SkippedDecode", "Reset"), Transitions),
        (("Finishing", "Reset"), Transitions),
        (("Finished", "Reset"), Transitions),
        (("Preempted", "Reset"), Transitions),
        // Preempt
        (("Finished", "Preempt"), NoOp),
        (("Preempted", "Preempt"), NoOp),
        (("LookingUp", "Preempt"), Transitions),
        (("Prefetching", "Preempt"), Transitions),
        (("OnboardReady", "Preempt"), Transitions),
        (("Initialized", "Preempt"), Transitions),
        (("AwaitingLookup", "Preempt"), Transitions),
        (("Onboarding", "Preempt"), Transitions),
        (("Prefilling", "Preempt"), Transitions),
        (("SkippedPrefill", "Preempt"), Transitions),
        (("Decoding", "Preempt"), Transitions),
        (("SkippedDecode", "Preempt"), Transitions),
        (("Finishing", "Preempt"), Transitions),
        // Initialized
        (("Initialized", "AcquireMatches"), Transitions),
        (("Initialized", "ApplySchedulerOutput"), DataDependent),
        (("Initialized", "LocalLookupCompleted"), NoOp),
        (("Initialized", "RemoteLookupCompleted"), NoOp),
        (("Initialized", "RemoteLookupClosed"), NoOp),
        (("Initialized", "PrefetchReady"), NoOp),
        (("Initialized", "PrefetchTimeout"), NoOp),
        (("Initialized", "PrefetchFailed"), NoOp),
        (("Initialized", "PollMatchReport"), NoOp),
        (("Initialized", "AllocCompleted"), NoOp),
        (("Initialized", "MarkSkipped"), NoOp),
        (("Initialized", "TransferDispatched"), NoOp),
        (("Initialized", "TransferCompleted"), NoOp),
        (("Initialized", "TransferFailed"), NoOp),
        (("Initialized", "RequestFinished"), NoOp),
        // AwaitingLookup
        (("AwaitingLookup", "LocalLookupCompleted"), DataDependent),
        (("AwaitingLookup", "AcquireMatches"), NoOp),
        (("AwaitingLookup", "RemoteLookupCompleted"), NoOp),
        (("AwaitingLookup", "RemoteLookupClosed"), NoOp),
        (("AwaitingLookup", "PrefetchReady"), NoOp),
        (("AwaitingLookup", "PrefetchTimeout"), NoOp),
        (("AwaitingLookup", "PrefetchFailed"), NoOp),
        (("AwaitingLookup", "PollMatchReport"), NoOp),
        (("AwaitingLookup", "AllocCompleted"), NoOp),
        (("AwaitingLookup", "ApplySchedulerOutput"), NoOp),
        (("AwaitingLookup", "MarkSkipped"), NoOp),
        (("AwaitingLookup", "TransferDispatched"), NoOp),
        (("AwaitingLookup", "TransferCompleted"), NoOp),
        (("AwaitingLookup", "TransferFailed"), NoOp),
        (("AwaitingLookup", "RequestFinished"), NoOp),
        // LookingUp
        (("LookingUp", "RemoteLookupCompleted"), DataDependent),
        (("LookingUp", "RemoteLookupClosed"), DataDependent),
        (("LookingUp", "AcquireMatches"), NoOp),
        (("LookingUp", "LocalLookupCompleted"), NoOp),
        (("LookingUp", "PrefetchReady"), NoOp),
        (("LookingUp", "PrefetchTimeout"), NoOp),
        (("LookingUp", "PrefetchFailed"), NoOp),
        (("LookingUp", "PollMatchReport"), NoOp),
        (("LookingUp", "AllocCompleted"), NoOp),
        (("LookingUp", "ApplySchedulerOutput"), NoOp),
        (("LookingUp", "MarkSkipped"), NoOp),
        (("LookingUp", "TransferDispatched"), NoOp),
        (("LookingUp", "TransferCompleted"), NoOp),
        (("LookingUp", "TransferFailed"), NoOp),
        (("LookingUp", "RequestFinished"), NoOp),
        // Prefetching
        (("Prefetching", "PrefetchReady"), Transitions),
        (("Prefetching", "TransferCompleted"), DataDependent),
        (("Prefetching", "PrefetchTimeout"), Transitions),
        (("Prefetching", "PrefetchFailed"), DataDependent),
        (("Prefetching", "AcquireMatches"), NoOp),
        (("Prefetching", "LocalLookupCompleted"), NoOp),
        (("Prefetching", "RemoteLookupCompleted"), NoOp),
        (("Prefetching", "RemoteLookupClosed"), NoOp),
        (("Prefetching", "PollMatchReport"), NoOp),
        (("Prefetching", "AllocCompleted"), NoOp),
        (("Prefetching", "ApplySchedulerOutput"), NoOp),
        (("Prefetching", "MarkSkipped"), NoOp),
        (("Prefetching", "TransferDispatched"), NoOp),
        (("Prefetching", "TransferFailed"), NoOp),
        (("Prefetching", "RequestFinished"), NoOp),
        // OnboardReady
        (("OnboardReady", "PollMatchReport"), Transitions),
        (("OnboardReady", "AllocCompleted"), Transitions),
        (("OnboardReady", "AcquireMatches"), NoOp),
        (("OnboardReady", "LocalLookupCompleted"), NoOp),
        (("OnboardReady", "RemoteLookupCompleted"), NoOp),
        (("OnboardReady", "RemoteLookupClosed"), NoOp),
        (("OnboardReady", "PrefetchReady"), NoOp),
        (("OnboardReady", "PrefetchTimeout"), NoOp),
        (("OnboardReady", "PrefetchFailed"), NoOp),
        (("OnboardReady", "ApplySchedulerOutput"), NoOp),
        (("OnboardReady", "MarkSkipped"), NoOp),
        (("OnboardReady", "TransferDispatched"), NoOp),
        (("OnboardReady", "TransferCompleted"), NoOp),
        (("OnboardReady", "TransferFailed"), NoOp),
        (("OnboardReady", "RequestFinished"), NoOp),
        // Onboarding
        (("Onboarding", "ApplySchedulerOutput"), DataDependent),
        (("Onboarding", "AcquireMatches"), NoOp),
        (("Onboarding", "LocalLookupCompleted"), NoOp),
        (("Onboarding", "RemoteLookupCompleted"), NoOp),
        (("Onboarding", "RemoteLookupClosed"), NoOp),
        (("Onboarding", "PrefetchReady"), NoOp),
        (("Onboarding", "PrefetchTimeout"), NoOp),
        (("Onboarding", "PrefetchFailed"), NoOp),
        (("Onboarding", "PollMatchReport"), NoOp),
        (("Onboarding", "AllocCompleted"), NoOp),
        (("Onboarding", "MarkSkipped"), NoOp),
        (("Onboarding", "TransferDispatched"), NoOp),
        (("Onboarding", "TransferCompleted"), NoOp),
        (("Onboarding", "TransferFailed"), NoOp),
        (("Onboarding", "RequestFinished"), NoOp),
        // Prefilling
        (("Prefilling", "ApplySchedulerOutput"), DataDependent),
        (("Prefilling", "MarkSkipped"), Transitions),
        (("Prefilling", "AcquireMatches"), NoOp),
        (("Prefilling", "LocalLookupCompleted"), NoOp),
        (("Prefilling", "RemoteLookupCompleted"), NoOp),
        (("Prefilling", "RemoteLookupClosed"), NoOp),
        (("Prefilling", "PrefetchReady"), NoOp),
        (("Prefilling", "PrefetchTimeout"), NoOp),
        (("Prefilling", "PrefetchFailed"), NoOp),
        (("Prefilling", "PollMatchReport"), NoOp),
        (("Prefilling", "AllocCompleted"), NoOp),
        (("Prefilling", "TransferDispatched"), NoOp),
        (("Prefilling", "TransferCompleted"), NoOp),
        (("Prefilling", "TransferFailed"), NoOp),
        (("Prefilling", "RequestFinished"), Transitions),
        // SkippedPrefill
        (("SkippedPrefill", "ApplySchedulerOutput"), DataDependent),
        (("SkippedPrefill", "AcquireMatches"), NoOp),
        (("SkippedPrefill", "LocalLookupCompleted"), NoOp),
        (("SkippedPrefill", "RemoteLookupCompleted"), NoOp),
        (("SkippedPrefill", "RemoteLookupClosed"), NoOp),
        (("SkippedPrefill", "PrefetchReady"), NoOp),
        (("SkippedPrefill", "PrefetchTimeout"), NoOp),
        (("SkippedPrefill", "PrefetchFailed"), NoOp),
        (("SkippedPrefill", "PollMatchReport"), NoOp),
        (("SkippedPrefill", "AllocCompleted"), NoOp),
        (("SkippedPrefill", "MarkSkipped"), NoOp),
        (("SkippedPrefill", "TransferDispatched"), NoOp),
        (("SkippedPrefill", "TransferCompleted"), NoOp),
        (("SkippedPrefill", "TransferFailed"), NoOp),
        (("SkippedPrefill", "RequestFinished"), Transitions),
        // Decoding
        (("Decoding", "ApplySchedulerOutput"), NoOp),
        (("Decoding", "MarkSkipped"), Transitions),
        (("Decoding", "RequestFinished"), DataDependent),
        (("Decoding", "TransferCompleted"), Transitions),
        (("Decoding", "TransferFailed"), Transitions),
        (("Decoding", "TransferDispatched"), NoOp),
        (("Decoding", "AcquireMatches"), NoOp),
        (("Decoding", "LocalLookupCompleted"), NoOp),
        (("Decoding", "RemoteLookupCompleted"), NoOp),
        (("Decoding", "RemoteLookupClosed"), NoOp),
        (("Decoding", "PrefetchReady"), NoOp),
        (("Decoding", "PrefetchTimeout"), NoOp),
        (("Decoding", "PrefetchFailed"), NoOp),
        (("Decoding", "PollMatchReport"), NoOp),
        (("Decoding", "AllocCompleted"), NoOp),
        // SkippedDecode
        (("SkippedDecode", "ApplySchedulerOutput"), Transitions),
        (("SkippedDecode", "AcquireMatches"), NoOp),
        (("SkippedDecode", "LocalLookupCompleted"), NoOp),
        (("SkippedDecode", "RemoteLookupCompleted"), NoOp),
        (("SkippedDecode", "RemoteLookupClosed"), NoOp),
        (("SkippedDecode", "PrefetchReady"), NoOp),
        (("SkippedDecode", "PrefetchTimeout"), NoOp),
        (("SkippedDecode", "PrefetchFailed"), NoOp),
        (("SkippedDecode", "PollMatchReport"), NoOp),
        (("SkippedDecode", "AllocCompleted"), NoOp),
        (("SkippedDecode", "MarkSkipped"), NoOp),
        (("SkippedDecode", "TransferDispatched"), NoOp),
        (("SkippedDecode", "TransferCompleted"), NoOp),
        (("SkippedDecode", "TransferFailed"), NoOp),
        (("SkippedDecode", "RequestFinished"), NoOp),
        // Finishing
        (("Finishing", "TransferCompleted"), DataDependent),
        (("Finishing", "TransferFailed"), DataDependent),
        (("Finishing", "TransferDispatched"), NoOp),
        (("Finishing", "AcquireMatches"), NoOp),
        (("Finishing", "LocalLookupCompleted"), NoOp),
        (("Finishing", "RemoteLookupCompleted"), NoOp),
        (("Finishing", "RemoteLookupClosed"), NoOp),
        (("Finishing", "PrefetchReady"), NoOp),
        (("Finishing", "PrefetchTimeout"), NoOp),
        (("Finishing", "PrefetchFailed"), NoOp),
        (("Finishing", "PollMatchReport"), NoOp),
        (("Finishing", "AllocCompleted"), NoOp),
        (("Finishing", "ApplySchedulerOutput"), NoOp),
        (("Finishing", "MarkSkipped"), NoOp),
        (("Finishing", "RequestFinished"), NoOp),
        // Finished
        (("Finished", "AcquireMatches"), NoOp),
        (("Finished", "LocalLookupCompleted"), NoOp),
        (("Finished", "RemoteLookupCompleted"), NoOp),
        (("Finished", "RemoteLookupClosed"), NoOp),
        (("Finished", "PrefetchReady"), NoOp),
        (("Finished", "PrefetchTimeout"), NoOp),
        (("Finished", "PrefetchFailed"), NoOp),
        (("Finished", "PollMatchReport"), NoOp),
        (("Finished", "AllocCompleted"), NoOp),
        (("Finished", "ApplySchedulerOutput"), NoOp),
        (("Finished", "MarkSkipped"), NoOp),
        (("Finished", "TransferDispatched"), NoOp),
        (("Finished", "TransferCompleted"), NoOp),
        (("Finished", "TransferFailed"), NoOp),
        (("Finished", "RequestFinished"), NoOp),
        // Preempted
        (("Preempted", "AcquireMatches"), Transitions),
        (("Preempted", "LocalLookupCompleted"), NoOp),
        (("Preempted", "RemoteLookupCompleted"), NoOp),
        (("Preempted", "RemoteLookupClosed"), NoOp),
        (("Preempted", "PrefetchReady"), NoOp),
        (("Preempted", "PrefetchTimeout"), NoOp),
        (("Preempted", "PrefetchFailed"), NoOp),
        (("Preempted", "PollMatchReport"), NoOp),
        (("Preempted", "AllocCompleted"), NoOp),
        (("Preempted", "ApplySchedulerOutput"), NoOp),
        (("Preempted", "MarkSkipped"), NoOp),
        (("Preempted", "TransferDispatched"), NoOp),
        (("Preempted", "TransferCompleted"), NoOp),
        (("Preempted", "TransferFailed"), NoOp),
        (("Preempted", "RequestFinished"), NoOp),
    ];
    entries.into_iter().collect()
}

fn phase_name(phase: &TestPhase) -> &'static str {
    match phase {
        RequestPhase::Initialized => "Initialized",
        RequestPhase::AwaitingLookup => "AwaitingLookup",
        RequestPhase::LookingUp { .. } => "LookingUp",
        RequestPhase::Prefetching { .. } => "Prefetching",
        RequestPhase::OnboardReady { .. } => "OnboardReady",
        RequestPhase::Onboarding { .. } => "Onboarding",
        RequestPhase::Prefilling { .. } => "Prefilling",
        RequestPhase::SkippedPrefill => "SkippedPrefill",
        RequestPhase::Decoding { .. } => "Decoding",
        RequestPhase::SkippedDecode { .. } => "SkippedDecode",
        RequestPhase::Finishing { .. } => "Finishing",
        RequestPhase::Finished => "Finished",
        RequestPhase::Preempted { .. } => "Preempted",
    }
}

#[test]
fn exhaustive_transition_matrix() {
    let table = transition_table();

    assert_eq!(
        table.len(),
        ALL_PHASE_NAMES.len() * ALL_EVENT_NAMES.len(),
        "transition table must have exactly phases*events entries ({}*{}={}); got {}",
        ALL_PHASE_NAMES.len(), ALL_EVENT_NAMES.len(),
        ALL_PHASE_NAMES.len() * ALL_EVENT_NAMES.len(), table.len(),
    );

    let ctx = mock_ctx();
    let mut noop_tested = 0;

    for &p_name in ALL_PHASE_NAMES {
        for &e_name in ALL_EVENT_NAMES {
            let key = (p_name, e_name);
            let expectation = table.get(&key).unwrap_or_else(|| {
                panic!("MISSING from transition table: ({}, {})", p_name, e_name);
            });

            if *expectation == Expectation::NoOp {
                let phase = make_phase(p_name);
                let event = make_event(e_name);
                let (new_phase, effects) = phase.apply(event, &ctx);
                let new_name = phase_name(&new_phase);
                assert_eq!(
                    new_name, p_name,
                    "NoOp ({}, {}) changed phase from {} to {}",
                    p_name, e_name, p_name, new_name,
                );
                assert!(
                    effects.is_empty()
                        || effects.iter().all(|e| matches!(e, SlotEffect::Diag { .. })),
                    "NoOp ({}, {}) produced non-Diag effects: {:?}",
                    p_name, e_name, effects,
                );
                noop_tested += 1;
            }
        }
    }

    assert!(
        noop_tested > 100,
        "expected >100 NoOp pairs tested, got {}",
        noop_tested,
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Part C: Structural invariant checker
// ═══════════════════════════════════════════════════════════════════════════

fn check_invariants(phase: &TestPhase, block_size: usize) {
    match phase {
        RequestPhase::OnboardReady {
            num_external_tokens,
            host_staging,
            disk_staging,
            ..
        } => {
            let actual_blocks = host_staging.len() + disk_staging.len();
            assert!(
                actual_blocks > 0,
                "invariant: OnboardReady with zero staging blocks"
            );
            assert_eq!(
                *num_external_tokens,
                actual_blocks * block_size,
                "invariant: OnboardReady num_external_tokens ({}) != staging blocks ({}) * block_size ({})",
                num_external_tokens, actual_blocks, block_size,
            );
        }
        RequestPhase::Onboarding {
            num_external_tokens,
            ..
        } => {
            assert!(
                *num_external_tokens > 0,
                "invariant: Onboarding with zero external tokens"
            );
        }
        RequestPhase::Finishing { ops } => {
            assert!(
                ops.has_any(),
                "invariant: Finishing must have pending ops (otherwise should be Finished)"
            );
        }
        _ => {}
    }
}

fn apply_checked(
    phase: TestPhase,
    event: TestEvent,
    ctx: &SlotContext,
) -> (TestPhase, Vec<SlotEffect<(), ()>>) {
    let (new_phase, effects) = phase.apply(event, ctx);
    check_invariants(&new_phase, ctx.block_size);
    (new_phase, effects)
}

#[test]
fn invariant_checker_catches_phantom_onboard_ready() {
    let bad = RequestPhase::<(), ()>::OnboardReady {
        num_external_tokens: 512,
        host_staging: vec![],
        disk_staging: vec![],
        remote_hashes: vec![],
        disclosure: DisclosureState::Pending,
        prefetched_blocks_for_stats: 0,
    };
    let result = std::panic::catch_unwind(|| check_invariants(&bad, 256));
    assert!(result.is_err(), "invariant checker must catch zero-staging OnboardReady");
}

#[test]
fn invariant_checker_accepts_valid_onboard_ready() {
    let good = RequestPhase::<(), ()>::OnboardReady {
        num_external_tokens: 32,
        host_staging: vec![(), ()],
        disk_staging: vec![],
        remote_hashes: vec![],
        disclosure: DisclosureState::Pending,
        prefetched_blocks_for_stats: 0,
    };
    check_invariants(&good, 16);
}
