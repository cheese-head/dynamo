// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared test harness for Tier 2 and Tier 3 KVBM tests.
//!
//! Provides builders and helpers for constructing test instances of
//! ConnectorSlotManager, OperationTracker, and related types without
//! requiring the full distributed runtime.

#[cfg(test)]
pub(crate) mod helpers {
    use crate::block_manager::connector::protocol::{
        RequestType, SlotKey, TransferType, WorkerTransferRequest,
    };
    use crate::block_manager::distributed::vllm::slot_ops::OperationTracker;
    use crate::tokens::{TokenBlock, TokenBlockSequence, Tokens};

    /// Create a SlotKey for testing.
    pub fn test_slot_key(req_id: &str, generation: u64) -> SlotKey {
        SlotKey::new(req_id.to_string(), generation)
    }

    /// Create a mock WorkerTransferRequest.
    pub fn mock_worker_request(
        req_id: &str,
        transfer_type: TransferType,
        block_ids: Vec<usize>,
    ) -> WorkerTransferRequest {
        WorkerTransferRequest {
            key: test_slot_key(req_id, 0),
            uuid: uuid::Uuid::new_v4(),
            transfer_type,
            request_type: RequestType::Immediate,
            block_ids,
        }
    }

    /// Create a mock load request.
    pub fn mock_load_request(req_id: &str, block_ids: Vec<usize>) -> WorkerTransferRequest {
        mock_worker_request(req_id, TransferType::Load, block_ids)
    }

    /// Create a mock store request.
    ///
    /// Used by `e2e_tests` when `testing-nixl` + `testing-cuda` are enabled; otherwise unused
    /// in default `cargo test --no-run` builds.
    #[allow(dead_code)]
    pub fn mock_store_request(req_id: &str, block_ids: Vec<usize>) -> WorkerTransferRequest {
        mock_worker_request(req_id, TransferType::Store, block_ids)
    }

    /// Create token blocks from raw token IDs (block_size=1 for simplicity).
    ///
    /// Used by `e2e_tests` when `testing-nixl` + `testing-cuda` are enabled; otherwise
    /// unused in default `cargo test` builds.
    #[allow(dead_code)]
    pub fn make_token_blocks(tokens: &[u32]) -> Vec<TokenBlock> {
        TokenBlockSequence::new(Tokens::from(tokens), 1, Some(0))
            .blocks()
            .to_vec()
    }

    /// Create an OperationTracker pre-loaded with N pending load operations.
    pub fn tracker_with_pending_loads(req_id: &str, n: usize) -> OperationTracker {
        let mut tracker = OperationTracker::new();
        for i in 0..n {
            tracker.append_pending(mock_load_request(req_id, vec![i]));
        }
        tracker
    }

    /// Create an OperationTracker with operations already dispatched.
    pub fn tracker_with_dispatched(req_id: &str, n: usize) -> OperationTracker {
        let mut tracker = tracker_with_pending_loads(req_id, n);
        let _ = tracker.take_pending_for_dispatch();
        tracker
    }
}
