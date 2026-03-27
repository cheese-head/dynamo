// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Tier-1 pure-logic unit tests for the KVBM slot state machine.
//! No I/O, no NIXL, no GPU — tests data structures, serialization, and state logic.

use rstest::rstest;

use crate::block_manager::{
    connector::protocol::{RequestType, SlotKey, TransferType, WorkerTransferRequest},
    distributed::vllm::{ConnectorMetadata, NewSlotKind, OperationTracker},
};

// ---------------------------------------------------------------------------
// Category 1: OperationTracker lifecycle
// ---------------------------------------------------------------------------

#[test]
fn test_operation_tracker_empty_initially() {
    let tracker = OperationTracker::new();
    assert!(!tracker.has_any());
    assert_eq!(tracker.pending_count(), 0);
    assert_eq!(tracker.dispatched_count(), 0);
}

#[test]
fn test_operation_tracker_append_and_dispatch() {
    let mut tracker = OperationTracker::new();

    let op1 = WorkerTransferRequest {
        key: SlotKey::new("req-1".into(), 0),
        uuid: uuid::Uuid::new_v4(),
        transfer_type: TransferType::Load,
        request_type: RequestType::Immediate,
        block_ids: vec![],
    };
    let op2 = WorkerTransferRequest {
        key: SlotKey::new("req-1".into(), 0),
        uuid: uuid::Uuid::new_v4(),
        transfer_type: TransferType::Store,
        request_type: RequestType::Scheduled,
        block_ids: vec![1, 2],
    };

    tracker.append_pending(op1);
    tracker.append_pending(op2);
    assert_eq!(tracker.pending_count(), 2);
    assert_eq!(tracker.dispatched_count(), 0);
    assert!(tracker.has_any());

    let taken = tracker.take_pending_for_dispatch();
    assert!(taken.is_some());
    assert_eq!(taken.unwrap().len(), 2);
    assert_eq!(tracker.pending_count(), 0);
    assert_eq!(tracker.dispatched_count(), 2);
    assert!(tracker.has_any());
}

#[test]
fn test_operation_tracker_discard_pending() {
    let mut tracker = OperationTracker::new();

    let op = WorkerTransferRequest {
        key: SlotKey::new("req-2".into(), 0),
        uuid: uuid::Uuid::new_v4(),
        transfer_type: TransferType::Load,
        request_type: RequestType::Immediate,
        block_ids: vec![],
    };
    tracker.append_pending(op);
    assert_eq!(tracker.pending_count(), 1);

    let discarded = tracker.discard_pending();
    assert_eq!(discarded, 1);
    assert!(!tracker.has_any());
    assert_eq!(tracker.pending_count(), 0);
    assert_eq!(tracker.dispatched_count(), 0);
}

#[test]
fn test_operation_tracker_clear_all() {
    let mut tracker = OperationTracker::new();

    let op = WorkerTransferRequest {
        key: SlotKey::new("req-3".into(), 0),
        uuid: uuid::Uuid::new_v4(),
        transfer_type: TransferType::Load,
        request_type: RequestType::Immediate,
        block_ids: vec![],
    };
    tracker.append_pending(op);
    tracker.take_pending_for_dispatch();
    assert!(tracker.has_any());

    tracker.clear_all();
    assert!(!tracker.has_any());
    assert_eq!(tracker.pending_count(), 0);
    assert_eq!(tracker.dispatched_count(), 0);
}

#[test]
fn test_operation_tracker_take_pending_when_none() {
    let mut tracker = OperationTracker::new();
    assert!(tracker.take_pending_for_dispatch().is_none());
    assert_eq!(tracker.dispatched_count(), 0);
}

#[test]
fn test_operation_tracker_discard_when_none() {
    let mut tracker = OperationTracker::new();
    assert_eq!(tracker.discard_pending(), 0);
}

// ---------------------------------------------------------------------------
// Category 4: ConnectorMetadata serialization roundtrip
// ---------------------------------------------------------------------------

#[test]
fn test_connector_metadata_serde_roundtrip() {
    let mut meta = ConnectorMetadata::new(42);
    meta.create_onboarding_slot(SlotKey::new("req-meta-1".into(), 3), 2);
    meta.operations.push(WorkerTransferRequest {
        key: SlotKey::new("req-meta-1".into(), 3),
        uuid: uuid::Uuid::new_v4(),
        transfer_type: TransferType::Load,
        request_type: RequestType::Immediate,
        block_ids: vec![10, 20],
    });

    let bytes = serde_json::to_vec(&meta).unwrap();
    let deserialized: ConnectorMetadata = serde_json::from_slice(&bytes).unwrap();

    assert_eq!(deserialized.iteration, 42);
    assert_eq!(deserialized.new_slots.len(), 1);
    assert_eq!(deserialized.new_slots[0].key.request_id, "req-meta-1");
    assert!(matches!(
        deserialized.new_slots[0].kind,
        NewSlotKind::Onboarding
    ));
    assert_eq!(deserialized.new_slots[0].expected_immediate_ops, 2);
    assert_eq!(deserialized.operations.len(), 1);
    assert_eq!(deserialized.operations[0].block_ids, vec![10, 20]);
}

#[test]
fn test_connector_metadata_empty_roundtrip() {
    let meta = ConnectorMetadata::new(0);
    let bytes = serde_json::to_vec(&meta).unwrap();
    let deserialized: ConnectorMetadata = serde_json::from_slice(&bytes).unwrap();

    assert_eq!(deserialized.iteration, 0);
    assert!(deserialized.new_slots.is_empty());
    assert!(deserialized.operations.is_empty());
}

// ---------------------------------------------------------------------------
// Category 5: SlotKey equality and display
// ---------------------------------------------------------------------------

#[test]
fn test_slot_key_equality() {
    let k1 = SlotKey::new("req-1".into(), 0);
    let k2 = SlotKey::new("req-1".into(), 0);
    let k3 = SlotKey::new("req-1".into(), 1);
    let k4 = SlotKey::new("req-2".into(), 0);

    assert_eq!(k1, k2);
    assert_ne!(k1, k3);
    assert_ne!(k1, k4);
}

#[test]
fn test_slot_key_display() {
    let key = SlotKey::new("req-abc".into(), 7);
    assert_eq!(format!("{}", key), "req-abc:7");
}

#[test]
fn test_slot_key_display_generation_zero() {
    let key = SlotKey::new("r".into(), 0);
    assert_eq!(format!("{}", key), "r:0");
}

#[test]
fn test_slot_key_hash_consistency() {
    use std::collections::HashSet;
    let k1 = SlotKey::new("req-1".into(), 0);
    let k2 = SlotKey::new("req-1".into(), 0);

    let mut set = HashSet::new();
    set.insert(k1);
    assert!(set.contains(&k2));
}

#[test]
fn test_slot_key_serde_roundtrip() {
    let key = SlotKey::new("req-serde".into(), 99);
    let bytes = serde_json::to_vec(&key).unwrap();
    let deserialized: SlotKey = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(deserialized, key);
}

// ---------------------------------------------------------------------------
// Category 6: WorkerTransferRequest serialization
// ---------------------------------------------------------------------------

#[test]
fn test_worker_transfer_request_serde() {
    let req = WorkerTransferRequest {
        key: SlotKey::new("req-1".into(), 0),
        uuid: uuid::Uuid::new_v4(),
        transfer_type: TransferType::Load,
        request_type: RequestType::Immediate,
        block_ids: vec![10, 20, 30],
    };
    let bytes = serde_json::to_vec(&req).unwrap();
    let deserialized: WorkerTransferRequest = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(deserialized.key, req.key);
    assert_eq!(deserialized.uuid, req.uuid);
    assert_eq!(deserialized.transfer_type, TransferType::Load);
    assert_eq!(deserialized.request_type, RequestType::Immediate);
    assert_eq!(deserialized.block_ids, vec![10, 20, 30]);
}

#[test]
fn test_worker_transfer_request_backward_compat_no_block_ids() {
    let json = r#"{"key":{"request_id":"req","generation":0},"uuid":"550e8400-e29b-41d4-a716-446655440000","transfer_type":"Load","request_type":"Immediate"}"#;
    let req: WorkerTransferRequest = serde_json::from_str(json).unwrap();
    assert!(req.block_ids.is_empty());
    assert_eq!(req.key.request_id, "req");
    assert_eq!(req.key.generation, 0);
    assert_eq!(req.transfer_type, TransferType::Load);
}

#[rstest]
#[case(TransferType::Load, "Load")]
#[case(TransferType::Store, "Store")]
fn test_transfer_type_serde_roundtrip(#[case] tt: TransferType, #[case] expected_str: &str) {
    let json = serde_json::to_string(&tt).unwrap();
    assert_eq!(json, format!("\"{}\"", expected_str));
    let deserialized: TransferType = serde_json::from_str(&json).unwrap();
    assert_eq!(deserialized, tt);
}

#[rstest]
#[case(RequestType::Immediate, "Immediate")]
#[case(RequestType::Scheduled, "Scheduled")]
fn test_request_type_serde_roundtrip(#[case] rt: RequestType, #[case] expected_str: &str) {
    let json = serde_json::to_string(&rt).unwrap();
    assert_eq!(json, format!("\"{}\"", expected_str));
    let deserialized: RequestType = serde_json::from_str(&json).unwrap();
    assert_eq!(deserialized, rt);
}
