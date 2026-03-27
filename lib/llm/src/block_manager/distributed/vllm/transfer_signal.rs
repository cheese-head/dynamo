// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Single source of truth for transfer operation completion.
//!
//! [`TransferSignal`] is a trait that producers (transfer engine) write to and
//! consumers (leader, worker) poll from. The default [`AtomicTransferSignal`]
//! uses `DashMap` + `AtomicU8` for lock-free, non-blocking reads with no
//! channels and no async/sync boundary crossings.

use std::collections::HashSet;
use std::sync::atomic::{AtomicU8, Ordering};

use dashmap::DashMap;
use uuid::Uuid;

use crate::block_manager::connector::protocol::TransferType;

// ---------------------------------------------------------------------------
// Status constants
// ---------------------------------------------------------------------------

const STATUS_PENDING: u8 = 0;
const STATUS_COMPLETE: u8 = 1;
const STATUS_FAILED: u8 = 2;
const STATUS_TAKEN: u8 = 3;

// ---------------------------------------------------------------------------
// Trait
// ---------------------------------------------------------------------------

/// Single source of truth for transfer operation completion.
///
/// Producers call [`register`], [`complete`], and [`fail`].
/// Consumers call [`is_loads_done`], [`is_stores_done`], [`has_failed`],
/// and [`take_completed`]. All methods are non-blocking.
pub trait TransferSignal: Send + Sync + 'static {
    /// Register a new pending operation. Called by the effect executor when
    /// dispatching onboard (Load) or offload (Store) transfers.
    fn register(&self, op_id: Uuid, request_id: &str, transfer_type: TransferType);

    /// Mark an operation as successfully completed. Called by the transfer
    /// engine when a DMA or disk transfer finishes.
    fn complete(&self, op_id: Uuid);

    /// Mark an operation as failed. Called by the transfer engine on error.
    fn fail(&self, op_id: Uuid);

    /// Are all Load (onboard/H2D) operations for this request done (Complete
    /// or Failed)? Returns `None` if no Load ops are registered.
    fn is_loads_done(&self, request_id: &str) -> Option<bool>;

    /// Are all Store (offload) operations for this request done (Complete or
    /// Failed)? Returns `None` if no Store ops are registered.
    fn is_stores_done(&self, request_id: &str) -> Option<bool>;

    /// Does this request have any failed operations?
    fn has_failed(&self, request_id: &str) -> bool;

    /// Take completed operations for a request (each op returned at most
    /// once). Used by the worker meta builder to report completions upstream.
    fn take_completed(&self, request_id: &str) -> Vec<(Uuid, TransferType)>;

    /// Remove all state for a request. Called after vLLM frees blocks.
    fn remove(&self, request_id: &str);

    /// Remove all state for all requests. Called on pool reset.
    fn clear(&self);
}

// ---------------------------------------------------------------------------
// AtomicTransferSignal — default lock-free implementation
// ---------------------------------------------------------------------------

struct OperationRecord {
    request_id: String,
    transfer_type: TransferType,
    status: AtomicU8,
}

/// Lock-free [`TransferSignal`] backed by `DashMap` + `AtomicU8`.
///
/// - **Write** (`complete`/`fail`): single atomic store.
/// - **Read** (`is_loads_done`/`is_stores_done`): iterate ops for request,
///   filter by type, read atomics. No locks held across the iteration
///   (DashMap read shards only).
/// - **Take** (`take_completed`): CAS from Complete→Taken so each op is
///   reported exactly once.
pub struct AtomicTransferSignal {
    operations: DashMap<Uuid, OperationRecord>,
    by_request: DashMap<String, HashSet<Uuid>>,
}

impl AtomicTransferSignal {
    pub fn new() -> Self {
        Self {
            operations: DashMap::new(),
            by_request: DashMap::new(),
        }
    }

    fn is_type_done(&self, request_id: &str, tt: TransferType) -> Option<bool> {
        let op_ids = self.by_request.get(request_id)?;
        let mut found_any = false;
        for op_id in op_ids.iter() {
            if let Some(rec) = self.operations.get(op_id) {
                if rec.transfer_type == tt {
                    found_any = true;
                    let s = rec.status.load(Ordering::Acquire);
                    if s == STATUS_PENDING {
                        return Some(false);
                    }
                }
            }
        }
        if found_any {
            Some(true)
        } else {
            None
        }
    }
}

impl Default for AtomicTransferSignal {
    fn default() -> Self {
        Self::new()
    }
}

impl TransferSignal for AtomicTransferSignal {
    fn register(&self, op_id: Uuid, request_id: &str, transfer_type: TransferType) {
        self.operations.insert(
            op_id,
            OperationRecord {
                request_id: request_id.to_string(),
                transfer_type,
                status: AtomicU8::new(STATUS_PENDING),
            },
        );
        self.by_request
            .entry(request_id.to_string())
            .or_default()
            .insert(op_id);
    }

    fn complete(&self, op_id: Uuid) {
        if let Some(rec) = self.operations.get(&op_id) {
            rec.status.store(STATUS_COMPLETE, Ordering::Release);
        }
    }

    fn fail(&self, op_id: Uuid) {
        if let Some(rec) = self.operations.get(&op_id) {
            rec.status.store(STATUS_FAILED, Ordering::Release);
        }
    }

    fn is_loads_done(&self, request_id: &str) -> Option<bool> {
        self.is_type_done(request_id, TransferType::Load)
    }

    fn is_stores_done(&self, request_id: &str) -> Option<bool> {
        self.is_type_done(request_id, TransferType::Store)
    }

    fn has_failed(&self, request_id: &str) -> bool {
        let Some(op_ids) = self.by_request.get(request_id) else {
            return false;
        };
        for op_id in op_ids.iter() {
            if let Some(rec) = self.operations.get(op_id) {
                if rec.status.load(Ordering::Acquire) == STATUS_FAILED {
                    return true;
                }
            }
        }
        false
    }

    fn take_completed(&self, request_id: &str) -> Vec<(Uuid, TransferType)> {
        let Some(op_ids) = self.by_request.get(request_id) else {
            return Vec::new();
        };
        let mut out = Vec::new();
        for &op_id in op_ids.iter() {
            if let Some(rec) = self.operations.get(&op_id) {
                let prev = rec
                    .status
                    .compare_exchange(
                        STATUS_COMPLETE,
                        STATUS_TAKEN,
                        Ordering::AcqRel,
                        Ordering::Acquire,
                    );
                if prev == Ok(STATUS_COMPLETE) {
                    out.push((op_id, rec.transfer_type));
                }
            }
        }
        out
    }

    fn remove(&self, request_id: &str) {
        if let Some((_, op_ids)) = self.by_request.remove(request_id) {
            for op_id in op_ids {
                self.operations.remove(&op_id);
            }
        }
    }

    fn clear(&self) {
        self.operations.clear();
        self.by_request.clear();
    }
}

// ---------------------------------------------------------------------------
// MockTransferSignal — for testing
// ---------------------------------------------------------------------------

#[cfg(any(test, feature = "testing"))]
pub mod mock {
    use super::*;
    use std::sync::Mutex;

    #[derive(Default)]
    pub struct MockTransferSignal {
        inner: AtomicTransferSignal,
        pub registered: Mutex<Vec<(Uuid, String, TransferType)>>,
    }

    impl MockTransferSignal {
        pub fn new() -> Self {
            Self::default()
        }

        pub fn script_complete(&self, op_id: Uuid) {
            self.inner.complete(op_id);
        }

        pub fn script_fail(&self, op_id: Uuid) {
            self.inner.fail(op_id);
        }
    }

    impl TransferSignal for MockTransferSignal {
        fn register(&self, op_id: Uuid, request_id: &str, transfer_type: TransferType) {
            if let Ok(mut v) = self.registered.lock() {
                v.push((op_id, request_id.to_string(), transfer_type));
            }
            self.inner.register(op_id, request_id, transfer_type);
        }

        fn complete(&self, op_id: Uuid) {
            self.inner.complete(op_id);
        }

        fn fail(&self, op_id: Uuid) {
            self.inner.fail(op_id);
        }

        fn is_loads_done(&self, request_id: &str) -> Option<bool> {
            self.inner.is_loads_done(request_id)
        }

        fn is_stores_done(&self, request_id: &str) -> Option<bool> {
            self.inner.is_stores_done(request_id)
        }

        fn has_failed(&self, request_id: &str) -> bool {
            self.inner.has_failed(request_id)
        }

        fn take_completed(&self, request_id: &str) -> Vec<(Uuid, TransferType)> {
            self.inner.take_completed(request_id)
        }

        fn remove(&self, request_id: &str) {
            self.inner.remove(request_id);
        }

        fn clear(&self) {
            self.inner.clear();
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn register_complete_query() {
        let sig = AtomicTransferSignal::new();
        let op = Uuid::new_v4();
        sig.register(op, "req-1", TransferType::Load);

        assert_eq!(sig.is_loads_done("req-1"), Some(false));
        assert_eq!(sig.is_stores_done("req-1"), None);

        sig.complete(op);
        assert_eq!(sig.is_loads_done("req-1"), Some(true));
        assert!(!sig.has_failed("req-1"));
    }

    #[test]
    fn fail_marks_failed() {
        let sig = AtomicTransferSignal::new();
        let op = Uuid::new_v4();
        sig.register(op, "req-1", TransferType::Load);

        sig.fail(op);
        assert_eq!(sig.is_loads_done("req-1"), Some(true));
        assert!(sig.has_failed("req-1"));
    }

    #[test]
    fn multiple_ops_per_request() {
        let sig = AtomicTransferSignal::new();
        let op1 = Uuid::new_v4();
        let op2 = Uuid::new_v4();
        sig.register(op1, "req-1", TransferType::Load);
        sig.register(op2, "req-1", TransferType::Load);

        assert_eq!(sig.is_loads_done("req-1"), Some(false));

        sig.complete(op1);
        assert_eq!(sig.is_loads_done("req-1"), Some(false));

        sig.complete(op2);
        assert_eq!(sig.is_loads_done("req-1"), Some(true));
    }

    #[test]
    fn mixed_load_store() {
        let sig = AtomicTransferSignal::new();
        let load_op = Uuid::new_v4();
        let store_op = Uuid::new_v4();
        sig.register(load_op, "req-1", TransferType::Load);
        sig.register(store_op, "req-1", TransferType::Store);

        assert_eq!(sig.is_loads_done("req-1"), Some(false));
        assert_eq!(sig.is_stores_done("req-1"), Some(false));

        sig.complete(load_op);
        assert_eq!(sig.is_loads_done("req-1"), Some(true));
        assert_eq!(sig.is_stores_done("req-1"), Some(false));

        sig.complete(store_op);
        assert_eq!(sig.is_stores_done("req-1"), Some(true));
    }

    #[test]
    fn take_completed_returns_each_once() {
        let sig = AtomicTransferSignal::new();
        let op = Uuid::new_v4();
        sig.register(op, "req-1", TransferType::Load);
        sig.complete(op);

        let first = sig.take_completed("req-1");
        assert_eq!(first.len(), 1);
        assert_eq!(first[0], (op, TransferType::Load));

        let second = sig.take_completed("req-1");
        assert!(second.is_empty());
    }

    #[test]
    fn remove_cleans_up() {
        let sig = AtomicTransferSignal::new();
        let op = Uuid::new_v4();
        sig.register(op, "req-1", TransferType::Load);
        sig.complete(op);

        sig.remove("req-1");
        assert_eq!(sig.is_loads_done("req-1"), None);
        assert!(sig.take_completed("req-1").is_empty());
    }

    #[test]
    fn unknown_request_returns_none() {
        let sig = AtomicTransferSignal::new();
        assert_eq!(sig.is_loads_done("nonexistent"), None);
        assert_eq!(sig.is_stores_done("nonexistent"), None);
        assert!(!sig.has_failed("nonexistent"));
    }

    #[test]
    fn complete_unknown_op_is_noop() {
        let sig = AtomicTransferSignal::new();
        sig.complete(Uuid::new_v4());
        sig.fail(Uuid::new_v4());
    }
}
