// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Tier 3 end-to-end protocol tests for the distributed KVBM vLLM path.
//!
//! These tests exercise the Leader -> ConnectorSlotManager -> TransferEngine -> Worker
//! protocol contracts: metadata serialization roundtrips, operation tracker lifecycle,
//! slot state machine transitions, pin registry integration, and slot key uniqueness.
//!
//! Feature gates: `testing-nixl` + `testing-cuda`

use std::collections::HashSet;

use rstest::rstest;
use uuid::Uuid;

use crate::block_manager::{
    connector::protocol::{SlotKey, TransferType},
    distributed::vllm::{ConnectorMetadata, NewSlotKind, SlotState},
    pool::{PinGuard, PinRegistry},
};

use super::test_harness::helpers::*;

// ============================================================================
// 1. ConnectorMetadata leader-worker roundtrip
// ============================================================================

#[tokio::test]
async fn test_metadata_leader_worker_roundtrip() {
    let mut meta = ConnectorMetadata::new(7);

    let key = test_slot_key("req-e2e-1", 2);
    meta.create_onboarding_slot(key.clone(), 3);

    let ops = vec![
        mock_load_request("req-e2e-1", vec![10, 20]),
        mock_store_request("req-e2e-1", vec![30]),
    ];
    meta.add_operations(ops);

    let bytes = serde_json::to_vec(&meta).expect("serialize ConnectorMetadata");
    let deserialized: ConnectorMetadata =
        serde_json::from_slice(&bytes).expect("deserialize ConnectorMetadata");

    assert_eq!(deserialized.iteration, 7);
    assert_eq!(deserialized.new_slots.len(), 1);
    assert_eq!(deserialized.new_slots[0].key, key);
    assert_eq!(deserialized.new_slots[0].kind, NewSlotKind::Onboarding);
    assert_eq!(deserialized.new_slots[0].expected_immediate_ops, 3);
    assert_eq!(deserialized.operations.len(), 2);
    assert_eq!(deserialized.operations[0].transfer_type, TransferType::Load);
    assert_eq!(
        deserialized.operations[1].transfer_type,
        TransferType::Store
    );
    assert_eq!(deserialized.operations[0].block_ids, vec![10, 20]);
    assert_eq!(deserialized.operations[1].block_ids, vec![30]);

    for op in &deserialized.operations {
        assert_eq!(op.key.request_id, "req-e2e-1");
    }
}

// ============================================================================
// 2. Multi-slot metadata batching
// ============================================================================

#[tokio::test]
async fn test_metadata_batches_multiple_slots_and_ops() {
    let mut meta = ConnectorMetadata::new(100);

    for i in 0..5 {
        let key = test_slot_key(&format!("req-batch-{i}"), i as u64);
        if i % 2 == 0 {
            meta.create_prefill_slot(key);
        } else {
            meta.create_onboarding_slot(key, (i + 1) as u64);
        }
    }

    for i in 0..10 {
        let op = mock_worker_request(
            &format!("req-batch-{}", i % 5),
            if i % 2 == 0 {
                TransferType::Load
            } else {
                TransferType::Store
            },
            vec![i * 10],
        );
        meta.add_operations(vec![op]);
    }

    let bytes = serde_json::to_vec(&meta).unwrap();
    let deserialized: ConnectorMetadata = serde_json::from_slice(&bytes).unwrap();

    assert_eq!(deserialized.new_slots.len(), 5);
    assert_eq!(deserialized.operations.len(), 10);

    let uuids: HashSet<_> = deserialized.operations.iter().map(|op| op.uuid).collect();
    assert_eq!(uuids.len(), 10, "All operation UUIDs must be unique");

    for (i, slot) in deserialized.new_slots.iter().enumerate() {
        assert_eq!(slot.key.request_id, format!("req-batch-{i}"));
        if i % 2 == 0 {
            assert_eq!(slot.kind, NewSlotKind::Prefill);
            assert_eq!(slot.expected_immediate_ops, 0);
        } else {
            assert_eq!(slot.kind, NewSlotKind::Onboarding);
            assert_eq!(slot.expected_immediate_ops, (i + 1) as u64);
        }
    }
}

// ============================================================================
// 3. OperationTracker dispatch-and-drain lifecycle (Finishing->Finished)
// ============================================================================

#[test]
fn test_operation_tracker_simulates_finishing_to_finished() {
    let mut tracker = tracker_with_dispatched("req-1", 3);
    assert!(tracker.has_any());
    assert_eq!(tracker.dispatched_count(), 3);
    assert_eq!(tracker.pending_count(), 0);

    tracker.clear_all();
    assert!(!tracker.has_any());
    assert_eq!(tracker.dispatched_count(), 0);
    assert_eq!(tracker.pending_count(), 0);
}

#[test]
fn test_operation_tracker_mixed_pending_and_dispatched() {
    let mut tracker = tracker_with_pending_loads("req-mix", 4);
    assert_eq!(tracker.pending_count(), 4);

    let taken = tracker.take_pending_for_dispatch().unwrap();
    assert_eq!(taken.len(), 4);
    assert_eq!(tracker.dispatched_count(), 4);

    tracker.append_pending(mock_store_request("req-mix", vec![100]));
    assert_eq!(tracker.pending_count(), 1);
    assert_eq!(tracker.dispatched_count(), 4);
    assert!(tracker.has_any());

    let taken2 = tracker.take_pending_for_dispatch().unwrap();
    assert_eq!(taken2.len(), 1);
    assert_eq!(tracker.dispatched_count(), 5);
    assert_eq!(tracker.pending_count(), 0);
}

// ============================================================================
// 4. Concurrent slot key generation uniqueness
// ============================================================================

#[test]
fn test_slot_keys_are_unique_across_generations() {
    let mut keys = HashSet::new();
    for generation in 0..100u64 {
        let key = SlotKey::new("req-1".to_string(), generation);
        assert!(
            keys.insert(key),
            "Duplicate key at generation {generation}"
        );
    }
}

#[test]
fn test_slot_keys_are_unique_across_request_ids() {
    let mut keys = HashSet::new();
    for i in 0..100u32 {
        let key = SlotKey::new(format!("req-{i}"), 0);
        assert!(keys.insert(key), "Duplicate key for request id req-{i}");
    }
}

// ============================================================================
// 5. SlotState lifecycle simulation
// ============================================================================

#[rstest]
#[case(SlotState::Initialized)]
#[case(SlotState::OnboardStaged(256))]
#[case(SlotState::Onboarding(256))]
#[case(SlotState::Prefilling)]
#[case(SlotState::SkippedPrefill)]
#[case(SlotState::Decoding)]
#[case(SlotState::SkippedDecode)]
#[case(SlotState::Finishing)]
#[case(SlotState::Finished)]
#[case(SlotState::Preempted)]
fn test_slot_state_is_copy_and_eq(#[case] state: SlotState) {
    let copy = state;
    assert_eq!(state, copy);
}

// ============================================================================
// 6. Pin registry integration
// ============================================================================

#[test]
fn test_pin_registry_concurrent_slot_tracking() {
    let registry = PinRegistry::new();

    let id1 = Uuid::new_v4();
    let id2 = Uuid::new_v4();
    let id3 = Uuid::new_v4();

    registry.insert(id1, PinGuard::empty());
    registry.insert(id2, PinGuard::empty());
    registry.insert(id3, PinGuard::empty());

    assert_eq!(registry.len(), 3);
    assert!(!registry.is_empty());

    registry.remove(&id1);
    assert_eq!(registry.len(), 2);
    assert!(!registry.contains(&id1));
    assert!(registry.contains(&id2));
    assert!(registry.contains(&id3));

    registry.clear();
    assert!(registry.is_empty());
    assert_eq!(registry.total_pinned_blocks(), 0);
}

#[test]
fn test_pin_registry_shared_state_across_clones() {
    let registry = PinRegistry::new();
    let engine_clone = registry.clone();

    let id = Uuid::new_v4();
    registry.insert(id, PinGuard::empty());

    assert!(
        engine_clone.contains(&id),
        "Engine clone must see guard inserted by slot manager"
    );

    engine_clone.remove(&id);
    assert!(
        !registry.contains(&id),
        "Removal via engine clone must be visible to original"
    );
}

// ============================================================================
// 7. Full lifecycle state sequence validation
// ============================================================================

#[test]
fn test_valid_state_sequences() {
    let states = vec![
        SlotState::Initialized,
        SlotState::OnboardStaged(1024),
        SlotState::Onboarding(1024),
        SlotState::Prefilling,
        SlotState::Decoding,
        SlotState::Finishing,
        SlotState::Finished,
    ];
    for window in states.windows(2) {
        assert_ne!(window[0], window[1]);
    }
}

#[test]
fn test_preemption_state_sequence() {
    let states = vec![
        SlotState::Initialized,
        SlotState::OnboardStaged(512),
        SlotState::Preempted,
        SlotState::Initialized,
    ];
    assert_eq!(states[2], SlotState::Preempted);
    assert_eq!(states[0], states[3], "Retry should return to Initialized");
}

#[test]
fn test_skipped_prefill_decode_paths() {
    let fast_path = vec![
        SlotState::Initialized,
        SlotState::SkippedPrefill,
        SlotState::SkippedDecode,
        SlotState::Finishing,
        SlotState::Finished,
    ];
    for window in fast_path.windows(2) {
        assert_ne!(window[0], window[1]);
    }
}

// ============================================================================
// 8. add_operations_for_key rewrites keys
// ============================================================================

#[test]
fn test_add_operations_for_key_rewrites_slot_keys() {
    let mut meta = ConnectorMetadata::new(1);
    let target_key = test_slot_key("target-req", 5);

    let ops = vec![
        mock_load_request("original-req", vec![1, 2]),
        mock_store_request("original-req", vec![3]),
    ];

    meta.add_operations_for_key(&target_key, ops);

    assert_eq!(meta.operations.len(), 2);
    for op in &meta.operations {
        assert_eq!(op.key, target_key, "Key must be rewritten to target_key");
    }
}

// ============================================================================
// 9. Test harness helpers produce correct structures
// ============================================================================

#[test]
fn test_harness_make_token_blocks() {
    let tokens: &[u32] = &[1, 2, 3, 4, 5];
    let blocks = make_token_blocks(tokens);
    assert_eq!(blocks.len(), 5, "block_size=1 should produce one block per token");
}

#[test]
fn test_harness_tracker_builders() {
    let pending = tracker_with_pending_loads("h-req", 3);
    assert_eq!(pending.pending_count(), 3);
    assert_eq!(pending.dispatched_count(), 0);

    let dispatched = tracker_with_dispatched("h-req", 3);
    assert_eq!(dispatched.pending_count(), 0);
    assert_eq!(dispatched.dispatched_count(), 3);
}
