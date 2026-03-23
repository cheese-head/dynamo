// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#[path = "kvbm_emulation/harness.rs"]
mod harness;
#[path = "kvbm_emulation/scenarios.rs"]
mod scenarios;

use std::time::Duration;

use dynamo_llm::block_manager::connector::protocol::SlotKey;
use harness::KvbmEmulationHarness;
use rstest::rstest;
use scenarios::load_fixture;

#[rstest]
#[case("fixture-basic-offload-onboard")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_offload_onboard_cycle_from_yaml_fixture(#[case] fixture_name: &str) {
    let fixture = load_fixture(fixture_name);
    let mut harness = KvbmEmulationHarness::new("worker-0").await;

    let key = SlotKey::new(fixture.request_id.clone(), fixture.generation);
    let counters = harness
        .bind_epoch(key.clone(), fixture.tp_bindings, fixture.immediate_ops)
        .await;

    harness
        .complete_immediate_many(key.clone(), fixture.immediate_ops)
        .await;
    harness
        .wait_for_all_bindings(
            &counters,
            fixture.immediate_ops as u64,
            Duration::from_secs(5),
        )
        .await;

    let (offload, onboard) = harness.build_offload_and_onboard_cycle(
        key.clone(),
        fixture.block_count,
        fixture.block_size,
    );

    assert_eq!(offload.key, key);
    assert_eq!(onboard.key, key);
    assert_eq!(offload.request_id, fixture.request_id);
    assert_eq!(onboard.request_id, fixture.request_id);
    assert_eq!(offload.block_ids.len(), fixture.block_count);
    assert_eq!(offload.block_size, fixture.block_size);
    assert_eq!(onboard.block_size, fixture.block_size);
    assert_eq!(offload.sequence_hashes, onboard.sequence_hashes);
    assert_eq!(onboard.device_block_ids, offload.block_ids);
    assert!(onboard.is_onboard);

    harness.finish_epoch(key, fixture.tp_bindings).await;
    harness.shutdown().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_stale_generation_completion_is_dropped_in_emulation_harness() {
    let mut harness = KvbmEmulationHarness::new("worker-0").await;

    let old_key = SlotKey::new("fixture-stale".to_string(), 0);
    let new_key = SlotKey::new("fixture-stale".to_string(), 1);

    let counters = harness.bind_epoch(new_key.clone(), 2, 2).await;
    harness.complete_immediate_many(old_key, 2).await;

    harness
        .wait_for_all_bindings(&counters, 0, Duration::from_millis(250))
        .await;

    harness.complete_immediate_many(new_key.clone(), 2).await;
    harness
        .wait_for_all_bindings(&counters, 2, Duration::from_secs(5))
        .await;

    harness.finish_epoch(new_key, 2).await;
    harness.shutdown().await;
}

#[rstest]
#[case(8, 2, 4)]
#[case(32, 4, 8)]
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn test_concurrent_kvbm_emulation_load(
    #[case] num_requests: usize,
    #[case] tp_bindings: usize,
    #[case] immediate_ops: usize,
) {
    let mut harness = KvbmEmulationHarness::new("worker-0").await;
    let mut all_counters = Vec::new();
    let mut keys = Vec::new();

    for request_index in 0..num_requests {
        let key = SlotKey::new(format!("load-{request_index}"), 0);
        let counters = harness
            .bind_epoch(key.clone(), tp_bindings, immediate_ops)
            .await;
        all_counters.push(counters);
        keys.push(key);
    }

    let mut tasks = Vec::new();
    for key in keys.iter().cloned() {
        let transfer_client = harness.transfer_client();
        tasks.push(tokio::spawn(async move {
            for _ in 0..immediate_ops {
                let request = harness::make_immediate_request(key.clone());
                let handle = transfer_client
                    .clone()
                    .schedule_transfer(request)
                    .await
                    .unwrap();
                handle.mark_complete(Ok(())).await;
            }
        }));
    }

    for task in tasks {
        task.await.unwrap();
    }

    for counters in &all_counters {
        harness
            .wait_for_all_bindings(counters, immediate_ops as u64, Duration::from_secs(10))
            .await;
    }

    for key in keys {
        harness.finish_epoch(key, tp_bindings).await;
    }
    harness.shutdown().await;
}
