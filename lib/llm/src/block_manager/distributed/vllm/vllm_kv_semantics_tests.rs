// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Mock vLLM v1 scheduler semantics around [`KVConnectorOutput`]-style recv/send
//! completion (`vllm/v1/core/sched/scheduler.py`: `_update_from_kv_xfer_finished`),
//! coordinated with our [`RequestPhase`] reducer.
//!
//! This does **not** import pyo3 or vLLM; it models the branch logic and assert
//! surface so regressions show up in `cargo test` without a Python stack.
//!
//! Covered (extend as you find bugs): wrong-state `finished_recving`, unknown ids,
//! batch recv/send, duplicate recv ids, recv-after-promote, `num_preemptions` →
//! `PREEMPTED` vs `WAITING` on promote, plain `Waiting`, `get_num_new_matched_tokens` /
//! async-load shape checks. Not modeled: `invalid_block_ids`, Python connector, full
//! `Scheduler.schedule()`.

#![cfg(test)]

use std::collections::{HashMap, HashSet};
use std::fmt;

use uuid::Uuid;

use super::slot_machine::{
    DisclosureState, G4FailPolicy, PrefetchState, RequestPhase, SlotContext, SlotEvent,
};

type TestPhase = RequestPhase<(), ()>;

// ---------------------------------------------------------------------------
// vLLM-shaped types (minimal)
// ---------------------------------------------------------------------------

/// Subset of `vllm.v1.request.RequestStatus` relevant to KV async recv.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VllmRequestStatus {
    Waiting,
    /// Async external KV load in flight (`WAITING_FOR_REMOTE_KVS`).
    WaitingForRemoteKvs,
    Running,
    Preempted,
    FinishedStopped,
}

impl VllmRequestStatus {
    fn is_finished(self) -> bool {
        matches!(self, VllmRequestStatus::FinishedStopped)
    }
}

#[derive(Debug, Clone, Default)]
struct VllmKvConnectorOutput {
    finished_recving: Vec<String>,
    finished_sending: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum VllmKvSemanticsError {
    /// `assert req_id in self.requests` (scheduler ~2100 / ~2109)
    UnknownRequestId {
        req_id: String,
        op: &'static str,
    },
    /// `assert RequestStatus.is_finished(req.status)` when not
    /// `WAITING_FOR_REMOTE_KVS` (~2105) — the production footgun for custom connectors.
    FinishedRecvingUnexpectedState {
        req_id: String,
        status: VllmRequestStatus,
    },
}

impl fmt::Display for VllmKvSemanticsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VllmKvSemanticsError::UnknownRequestId { req_id, op } => {
                write!(f, "unknown request_id {req_id} in {op}")
            }
            VllmKvSemanticsError::FinishedRecvingUnexpectedState { req_id, status } => {
                write!(
                    f,
                    "finished_recving for {req_id} while status {status:?} (vLLM expects WAITING_FOR_REMOTE_KVS or finished)"
                )
            }
        }
    }
}

impl std::error::Error for VllmKvSemanticsError {}

#[derive(Debug, Clone)]
struct VllmRequestRecord {
    status: VllmRequestStatus,
    /// Mirrors `Request.num_preemptions` — if > 0, vLLM promotes async KV recv to
    /// `PREEMPTED` instead of `WAITING` (`_try_promote_blocked_waiting_request`).
    num_preemptions: u32,
}

/// Minimal scheduler KV state: `finished_recving_kv_req_ids` and per-request status.
///
/// Mirrors `_update_from_kv_xfer_finished` branches; uses `Result` instead of Python
/// `assert` so tests can expect violations.
#[derive(Debug)]
struct VllmSchedulerKvMock {
    requests: HashMap<String, VllmRequestRecord>,
    finished_recving_kv_req_ids: HashSet<String>,
    /// How many times vLLM would call `_free_blocks` from this path.
    free_blocks_calls: usize,
}

impl VllmSchedulerKvMock {
    fn new() -> Self {
        Self {
            requests: HashMap::new(),
            finished_recving_kv_req_ids: HashSet::new(),
            free_blocks_calls: 0,
        }
    }

    fn insert_request(&mut self, req_id: impl Into<String>, status: VllmRequestStatus) {
        self.insert_request_with_preemptions(req_id, status, 0);
    }

    fn insert_request_with_preemptions(
        &mut self,
        req_id: impl Into<String>,
        status: VllmRequestStatus,
        num_preemptions: u32,
    ) {
        self.requests.insert(
            req_id.into(),
            VllmRequestRecord {
                status,
                num_preemptions,
            },
        );
    }

    fn status(&self, req_id: &str) -> Option<VllmRequestStatus> {
        self.requests.get(req_id).map(|r| r.status)
    }

    /// `Scheduler._update_from_kv_xfer_finished` (recv/send loops only).
    fn apply_kv_xfer_finished(&mut self, output: &VllmKvConnectorOutput) -> Result<(), VllmKvSemanticsError> {
        for req_id in &output.finished_recving {
            let Some(rec) = self.requests.get(req_id) else {
                return Err(VllmKvSemanticsError::UnknownRequestId {
                    req_id: req_id.clone(),
                    op: "finished_recving",
                });
            };
            match rec.status {
                VllmRequestStatus::WaitingForRemoteKvs => {
                    self.finished_recving_kv_req_ids.insert(req_id.clone());
                }
                s if s.is_finished() => {
                    self.free_blocks_calls += 1;
                }
                status => {
                    return Err(VllmKvSemanticsError::FinishedRecvingUnexpectedState {
                        req_id: req_id.clone(),
                        status,
                    });
                }
            }
        }

        for req_id in &output.finished_sending {
            if !self.requests.contains_key(req_id) {
                return Err(VllmKvSemanticsError::UnknownRequestId {
                    req_id: req_id.clone(),
                    op: "finished_sending",
                });
            }
            self.free_blocks_calls += 1;
        }

        Ok(())
    }

    /// `Scheduler._try_promote_blocked_waiting_request` for `WAITING_FOR_REMOTE_KVS`.
    fn try_promote_waiting_for_remote_kvs(&mut self, req_id: &str) -> bool {
        let Some(rec) = self.requests.get_mut(req_id) else {
            return false;
        };
        if rec.status != VllmRequestStatus::WaitingForRemoteKvs {
            return false;
        }
        if !self.finished_recving_kv_req_ids.contains(req_id) {
            return false;
        }
        if rec.num_preemptions > 0 {
            rec.status = VllmRequestStatus::Preempted;
        } else {
            rec.status = VllmRequestStatus::Waiting;
        }
        self.finished_recving_kv_req_ids.remove(req_id);
        true
    }
}

/// vLLM `load_kv_async` implies `num_external_computed_tokens > 0` (~648).
fn vllm_async_load_token_invariant(
    load_kv_async: bool,
    num_external_computed_tokens: usize,
) -> Result<(), &'static str> {
    if load_kv_async && num_external_computed_tokens == 0 {
        return Err("vLLM: load_kv_async requires num_external_computed_tokens > 0");
    }
    Ok(())
}

/// Contract for `KVConnectorBase_V1::get_num_new_matched_tokens` return shape
/// (scheduler uses `ext_tokens` + `load_kv_async` together).
fn vllm_get_num_new_matched_tokens_contract(
    ext_tokens: Option<usize>,
    load_kv_async: bool,
) -> Result<(), &'static str> {
    if load_kv_async {
        let Some(n) = ext_tokens else {
            return Err("vLLM: load_kv_async with ext_tokens=None is inconsistent");
        };
        if n == 0 {
            return Err("vLLM: load_kv_async requires ext_tokens > 0");
        }
    }
    Ok(())
}

fn mock_ctx() -> SlotContext {
    SlotContext {
        block_size: 16,
        remote_enabled: true,
        g4_xfer_fail_policy: G4FailPolicy::Fallback,
    }
}

// ---------------------------------------------------------------------------
// Tests: mock-only (vLLM semantics)
// ---------------------------------------------------------------------------

#[test]
fn vllm_mock_finished_recving_while_waiting_for_remote_ok() {
    let mut mock = VllmSchedulerKvMock::new();
    mock.insert_request("r1", VllmRequestStatus::WaitingForRemoteKvs);
    mock
        .apply_kv_xfer_finished(&VllmKvConnectorOutput {
            finished_recving: vec!["r1".into()],
            finished_sending: vec![],
        })
        .unwrap();
    assert!(mock.finished_recving_kv_req_ids.contains("r1"));
    assert!(mock.try_promote_waiting_for_remote_kvs("r1"));
    assert_eq!(mock.status("r1"), Some(VllmRequestStatus::Waiting));
    assert!(!mock.finished_recving_kv_req_ids.contains("r1"));
}

#[test]
fn vllm_mock_finished_recving_while_running_is_error() {
    let mut mock = VllmSchedulerKvMock::new();
    mock.insert_request("r1", VllmRequestStatus::Running);
    let err = mock
        .apply_kv_xfer_finished(&VllmKvConnectorOutput {
            finished_recving: vec!["r1".into()],
            finished_sending: vec![],
        })
        .unwrap_err();
    assert!(matches!(
        err,
        VllmKvSemanticsError::FinishedRecvingUnexpectedState { .. }
    ));
}

#[test]
fn vllm_mock_finished_recving_while_preempted_is_error() {
    // vLLM: `PREEMPTED` is not `WAITING_FOR_REMOTE_KVS` and not a finished status
    // (`is_finished` is `status > PREEMPTED`), so `finished_recving` hits the same
    // assert surface as `RUNNING`.
    let mut mock = VllmSchedulerKvMock::new();
    mock.insert_request("r1", VllmRequestStatus::Preempted);
    let err = mock
        .apply_kv_xfer_finished(&VllmKvConnectorOutput {
            finished_recving: vec!["r1".into()],
            finished_sending: vec![],
        })
        .unwrap_err();
    assert!(
        matches!(
            &err,
            VllmKvSemanticsError::FinishedRecvingUnexpectedState {
                status: VllmRequestStatus::Preempted,
                ..
            }
        ),
        "unexpected err: {err}"
    );
}

#[test]
fn vllm_mock_finished_recving_unknown_id_is_error() {
    let mut mock = VllmSchedulerKvMock::new();
    let err = mock
        .apply_kv_xfer_finished(&VllmKvConnectorOutput {
            finished_recving: vec!["ghost".into()],
            finished_sending: vec![],
        })
        .unwrap_err();
    assert!(matches!(
        err,
        VllmKvSemanticsError::UnknownRequestId { ref req_id, .. } if req_id == "ghost"
    ));
}

#[test]
fn vllm_mock_finished_recving_after_finished_frees_blocks() {
    let mut mock = VllmSchedulerKvMock::new();
    mock.insert_request("r1", VllmRequestStatus::FinishedStopped);
    mock
        .apply_kv_xfer_finished(&VllmKvConnectorOutput {
            finished_recving: vec!["r1".into()],
            finished_sending: vec![],
        })
        .unwrap();
    assert_eq!(mock.free_blocks_calls, 1);
}

#[test]
fn vllm_mock_finished_sending_unknown_id_is_error() {
    let mut mock = VllmSchedulerKvMock::new();
    let err = mock
        .apply_kv_xfer_finished(&VllmKvConnectorOutput {
            finished_recving: vec![],
            finished_sending: vec!["nope".into()],
        })
        .unwrap_err();
    assert!(matches!(
        err,
        VllmKvSemanticsError::UnknownRequestId { ref req_id, op: "finished_sending", .. }
            if req_id == "nope"
    ));
}

#[test]
fn vllm_async_load_invariant_rejects_zero_external_tokens() {
    assert!(vllm_async_load_token_invariant(true, 0).is_err());
    assert!(vllm_async_load_token_invariant(true, 1).is_ok());
    assert!(vllm_async_load_token_invariant(false, 0).is_ok());
}

#[test]
fn vllm_get_num_new_matched_tokens_contract_async_requires_some_positive() {
    assert!(vllm_get_num_new_matched_tokens_contract(None, true).is_err());
    assert!(vllm_get_num_new_matched_tokens_contract(Some(0), true).is_err());
    assert!(vllm_get_num_new_matched_tokens_contract(Some(1), true).is_ok());
    assert!(vllm_get_num_new_matched_tokens_contract(None, false).is_ok());
    assert!(vllm_get_num_new_matched_tokens_contract(Some(0), false).is_ok());
}

#[test]
fn vllm_mock_finished_recving_while_plain_waiting_is_error() {
    let mut mock = VllmSchedulerKvMock::new();
    mock.insert_request("r1", VllmRequestStatus::Waiting);
    let err = mock
        .apply_kv_xfer_finished(&VllmKvConnectorOutput {
            finished_recving: vec!["r1".into()],
            finished_sending: vec![],
        })
        .unwrap_err();
    assert!(
        matches!(
            &err,
            VllmKvSemanticsError::FinishedRecvingUnexpectedState {
                status: VllmRequestStatus::Waiting,
                ..
            }
        ),
        "{err}"
    );
}

#[test]
fn vllm_mock_batch_finished_recving_two_requests() {
    let mut mock = VllmSchedulerKvMock::new();
    mock.insert_request("a", VllmRequestStatus::WaitingForRemoteKvs);
    mock.insert_request("b", VllmRequestStatus::WaitingForRemoteKvs);
    mock
        .apply_kv_xfer_finished(&VllmKvConnectorOutput {
            finished_recving: vec!["a".into(), "b".into()],
            finished_sending: vec![],
        })
        .unwrap();
    assert!(mock.finished_recving_kv_req_ids.contains("a"));
    assert!(mock.finished_recving_kv_req_ids.contains("b"));
    assert!(mock.try_promote_waiting_for_remote_kvs("a"));
    assert!(mock.try_promote_waiting_for_remote_kvs("b"));
    assert_eq!(mock.status("a"), Some(VllmRequestStatus::Waiting));
    assert_eq!(mock.status("b"), Some(VllmRequestStatus::Waiting));
}

#[test]
fn vllm_mock_finished_sending_valid_id_increments_free_calls() {
    let mut mock = VllmSchedulerKvMock::new();
    mock.insert_request("s1", VllmRequestStatus::Running);
    mock
        .apply_kv_xfer_finished(&VllmKvConnectorOutput {
            finished_recving: vec![],
            finished_sending: vec!["s1".into()],
        })
        .unwrap();
    assert_eq!(mock.free_blocks_calls, 1);
}

#[test]
fn vllm_mock_recv_and_send_in_one_output() {
    let mut mock = VllmSchedulerKvMock::new();
    mock.insert_request("r", VllmRequestStatus::WaitingForRemoteKvs);
    mock.insert_request("s", VllmRequestStatus::FinishedStopped);
    mock
        .apply_kv_xfer_finished(&VllmKvConnectorOutput {
            finished_recving: vec!["r".into()],
            finished_sending: vec!["s".into()],
        })
        .unwrap();
    assert!(mock.finished_recving_kv_req_ids.contains("r"));
    assert_eq!(mock.free_blocks_calls, 1, "send on finished request frees");
}

#[test]
fn vllm_mock_promote_remote_kv_yields_preempted_when_num_preemptions_nonzero() {
    let mut mock = VllmSchedulerKvMock::new();
    mock.insert_request_with_preemptions("r1", VllmRequestStatus::WaitingForRemoteKvs, 2);
    mock
        .apply_kv_xfer_finished(&VllmKvConnectorOutput {
            finished_recving: vec!["r1".into()],
            finished_sending: vec![],
        })
        .unwrap();
    assert!(mock.try_promote_waiting_for_remote_kvs("r1"));
    assert_eq!(mock.status("r1"), Some(VllmRequestStatus::Preempted));
}

#[test]
fn vllm_mock_finished_recving_duplicate_batch_idempotent() {
    let mut mock = VllmSchedulerKvMock::new();
    mock.insert_request("r1", VllmRequestStatus::WaitingForRemoteKvs);
    mock
        .apply_kv_xfer_finished(&VllmKvConnectorOutput {
            finished_recving: vec!["r1".into(), "r1".into()],
            finished_sending: vec![],
        })
        .unwrap();
    assert_eq!(mock.finished_recving_kv_req_ids.len(), 1);
}

#[test]
fn vllm_mock_second_finished_recving_after_promote_errors() {
    let mut mock = VllmSchedulerKvMock::new();
    mock.insert_request("r1", VllmRequestStatus::WaitingForRemoteKvs);
    mock
        .apply_kv_xfer_finished(&VllmKvConnectorOutput {
            finished_recving: vec!["r1".into()],
            finished_sending: vec![],
        })
        .unwrap();
    assert!(mock.try_promote_waiting_for_remote_kvs("r1"));
    assert_eq!(mock.status("r1"), Some(VllmRequestStatus::Waiting));

    let err = mock
        .apply_kv_xfer_finished(&VllmKvConnectorOutput {
            finished_recving: vec!["r1".into()],
            finished_sending: vec![],
        })
        .unwrap_err();
    assert!(
        matches!(
            &err,
            VllmKvSemanticsError::FinishedRecvingUnexpectedState {
                status: VllmRequestStatus::Waiting,
                ..
            }
        ),
        "{err}"
    );
}

// ---------------------------------------------------------------------------
// Tests: slot machine + vLLM mock (same logical timestep)
// ---------------------------------------------------------------------------

#[test]
fn slot_prefetch_complete_and_vllm_finished_recving_stay_aligned() {
    const REQ: &str = "slot-req-1";
    let ctx = mock_ctx();
    let op_id = Uuid::new_v4();
    let prefetch = PrefetchState {
        operation_id: op_id,
        sequence_hashes: vec![0xabc],
        num_external_tokens: 32,
        started_at: std::time::Instant::now(),
    };
    let phase: TestPhase = RequestPhase::Prefetching {
        prefetch,
        host_staging: vec![()],
        disk_staging: vec![],
        prefetched_blocks_for_stats: 1,
    };

    let mut vllm = VllmSchedulerKvMock::new();
    vllm.insert_request(REQ, VllmRequestStatus::WaitingForRemoteKvs);

    let (new_phase, _) = phase.apply(SlotEvent::TransferCompleted { operation_id: op_id }, &ctx);
    assert!(
        matches!(new_phase, RequestPhase::OnboardReady { .. }),
        "Dynamo completes prefetch → OnboardReady"
    );

    vllm
        .apply_kv_xfer_finished(&VllmKvConnectorOutput {
            finished_recving: vec![REQ.into()],
            finished_sending: vec![],
        })
        .expect("vLLM accepts recv completion while WAITING_FOR_REMOTE_KVS");

    assert!(vllm.try_promote_waiting_for_remote_kvs(REQ));
    assert_eq!(vllm.status(REQ), Some(VllmRequestStatus::Waiting));
}

#[test]
fn desync_slot_onboard_but_vllm_still_waiting_for_remote_is_vllm_error() {
    const REQ: &str = "desync-1";
    let ctx = mock_ctx();
    let op_id = Uuid::new_v4();
    let prefetch = PrefetchState {
        operation_id: op_id,
        sequence_hashes: vec![],
        num_external_tokens: 16,
        started_at: std::time::Instant::now(),
    };
    let phase: TestPhase = RequestPhase::Prefetching {
        prefetch,
        host_staging: vec![],
        disk_staging: vec![],
        prefetched_blocks_for_stats: 0,
    };

    let (onboard_ready, _) =
        phase.apply(SlotEvent::TransferCompleted { operation_id: op_id }, &ctx);
    assert!(matches!(onboard_ready, RequestPhase::OnboardReady { .. }));

    // vLLM still thinks async load is in flight — reporting finished_recving is valid.
    let mut vllm = VllmSchedulerKvMock::new();
    vllm.insert_request(REQ, VllmRequestStatus::WaitingForRemoteKvs);
    vllm.apply_kv_xfer_finished(&VllmKvConnectorOutput {
        finished_recving: vec![REQ.into()],
        finished_sending: vec![],
    })
    .unwrap();

    // If vLLM were advanced to RUNNING without recv metadata alignment, recv complains:
    vllm.insert_request("bad", VllmRequestStatus::Running);
    let err = vllm
        .apply_kv_xfer_finished(&VllmKvConnectorOutput {
            finished_recving: vec!["bad".into()],
            finished_sending: vec![],
        })
        .unwrap_err();
    assert!(matches!(
        err,
        VllmKvSemanticsError::FinishedRecvingUnexpectedState { .. }
    ));
}

#[test]
fn slot_still_prefetching_while_vllm_reports_recv_is_consistent() {
    const REQ: &str = "inflight";
    let op_id = Uuid::new_v4();
    let phase: TestPhase = RequestPhase::Prefetching {
        prefetch: PrefetchState {
            operation_id: op_id,
            sequence_hashes: vec![1],
            num_external_tokens: 16,
            started_at: std::time::Instant::now(),
        },
        host_staging: vec![],
        disk_staging: vec![],
        prefetched_blocks_for_stats: 0,
    };

    let mut vllm = VllmSchedulerKvMock::new();
    vllm.insert_request(REQ, VllmRequestStatus::WaitingForRemoteKvs);

    vllm
        .apply_kv_xfer_finished(&VllmKvConnectorOutput {
            finished_recving: vec![REQ.into()],
            finished_sending: vec![],
        })
        .unwrap();

    assert!(
        matches!(phase, RequestPhase::Prefetching { .. }),
        "Dynamo can still be Prefetching until leader applies TransferCompleted"
    );
    assert!(vllm.finished_recving_kv_req_ids.contains(REQ));
}

#[test]
fn onboard_ready_matches_promoted_vllm_waiting_after_cache() {
    let ctx = mock_ctx();
    let op_id = Uuid::new_v4();
    let (phase, _): (TestPhase, _) = RequestPhase::Prefetching {
        prefetch: PrefetchState {
            operation_id: op_id,
            sequence_hashes: vec![0x55],
            num_external_tokens: 16,
            started_at: std::time::Instant::now(),
        },
        host_staging: vec![()],
        disk_staging: vec![],
        prefetched_blocks_for_stats: 2,
    }
    .apply(SlotEvent::TransferCompleted { operation_id: op_id }, &ctx);

    let RequestPhase::OnboardReady {
        num_external_tokens,
        remote_hashes,
        prefetched_blocks_for_stats,
        disclosure,
        ..
    } = phase
    else {
        panic!("expected OnboardReady");
    };
    assert_eq!(num_external_tokens, 16);
    assert_eq!(remote_hashes, vec![0x55]);
    assert_eq!(prefetched_blocks_for_stats, 2);
    assert_eq!(disclosure, DisclosureState::Pending);

    // vLLM side: after promotion, request is WAITING and will be scheduled RUNNING next.
    let mut vllm = VllmSchedulerKvMock::new();
    vllm.insert_request("r", VllmRequestStatus::WaitingForRemoteKvs);
    vllm
        .apply_kv_xfer_finished(&VllmKvConnectorOutput {
            finished_recving: vec!["r".into()],
            finished_sending: vec![],
        })
        .unwrap();
    assert!(vllm.try_promote_waiting_for_remote_kvs("r"));
    assert_eq!(vllm.status("r"), Some(VllmRequestStatus::Waiting));
}

#[test]
fn slot_onboard_and_vllm_promote_preempted_when_request_was_preempted_before() {
    const REQ: &str = "preempt-slot";
    let ctx = mock_ctx();
    let op_id = Uuid::new_v4();
    let phase: TestPhase = RequestPhase::Prefetching {
        prefetch: PrefetchState {
            operation_id: op_id,
            sequence_hashes: vec![0xaa],
            num_external_tokens: 64,
            started_at: std::time::Instant::now(),
        },
        host_staging: vec![(), ()],
        disk_staging: vec![],
        prefetched_blocks_for_stats: 2,
    };

    let mut vllm = VllmSchedulerKvMock::new();
    vllm.insert_request_with_preemptions(REQ, VllmRequestStatus::WaitingForRemoteKvs, 1);

    let (new_phase, _) = phase.apply(SlotEvent::TransferCompleted { operation_id: op_id }, &ctx);
    assert!(matches!(new_phase, RequestPhase::OnboardReady { .. }));

    vllm
        .apply_kv_xfer_finished(&VllmKvConnectorOutput {
            finished_recving: vec![REQ.into()],
            finished_sending: vec![],
        })
        .unwrap();
    assert!(vllm.try_promote_waiting_for_remote_kvs(REQ));
    assert_eq!(
        vllm.status(REQ),
        Some(VllmRequestStatus::Preempted),
        "vLLM resumes into PREEMPTED when num_preemptions > 0"
    );
}

// ---------------------------------------------------------------------------
// Hazard regressions (vLLM mock): stalls and ordering
// See `docs/kvbm-vllm-hazard-notes.md`.
// ---------------------------------------------------------------------------

#[test]
fn hazard_vllm_recv_recorded_but_no_promote_leaves_waiting_for_remote_kvs() {
    // H4: If the scheduler never runs promotion after `finished_recving`, the request
    // stays async-blocked forever from vLLM's point of view.
    let mut mock = VllmSchedulerKvMock::new();
    mock.insert_request("stuck", VllmRequestStatus::WaitingForRemoteKvs);
    mock
        .apply_kv_xfer_finished(&VllmKvConnectorOutput {
            finished_recving: vec!["stuck".into()],
            finished_sending: vec![],
        })
        .unwrap();
    assert!(mock.finished_recving_kv_req_ids.contains("stuck"));
    assert_eq!(
        mock.status("stuck"),
        Some(VllmRequestStatus::WaitingForRemoteKvs),
        "without try_promote, status must not advance"
    );
}

#[test]
fn hazard_vllm_promote_fails_without_prior_finished_recving() {
    let mut mock = VllmSchedulerKvMock::new();
    mock.insert_request("r", VllmRequestStatus::WaitingForRemoteKvs);
    assert!(
        !mock.try_promote_waiting_for_remote_kvs("r"),
        "promotion must not run before recv completion is recorded"
    );
    assert_eq!(mock.status("r"), Some(VllmRequestStatus::WaitingForRemoteKvs));
}

#[test]
fn hazard_vllm_async_contract_none_ext_tokens_is_reported_err_not_ok() {
    // H3: Connector must not tell vLLM load_kv_async without a positive token count.
    assert_eq!(
        vllm_get_num_new_matched_tokens_contract(None, true),
        Err("vLLM: load_kv_async with ext_tokens=None is inconsistent")
    );
}

/// H5: `WorkerFeedbackInstrumentation` records one count per dropped request batch (see `leader_core`).
#[test]
fn hazard_worker_feedback_unknown_req_id_increments_instrumentation() {
    use super::leader_core::WorkerFeedbackInstrumentation;

    let w = WorkerFeedbackInstrumentation::default();
    w.record_unknown_request("unregistered-req", 2, "completed_ops");
    assert_eq!(w.snapshot().unknown_request_id, 1);
    w.record_unknown_request("other", 1, "failed_ops");
    assert_eq!(w.snapshot().unknown_request_id, 2);
}
