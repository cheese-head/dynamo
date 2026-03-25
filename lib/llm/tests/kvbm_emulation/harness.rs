// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};
use std::time::{Duration, Instant};

use dynamo_llm::block_manager::connector::protocol::{LeaderTransferRequest, RequestType, SlotKey};
use dynamo_llm::block_manager::connector::scheduler::{
    Scheduler, SchedulerCreateSlotDetails, SchedulerMessage, SchedulerRemoveSlotDetails,
    TransferSchedulerClient,
};
use dynamo_llm::block_manager::distributed::vllm::{
    G4OnboardParams, LocalOffloadRequest, RemoteTransferRequest,
};
use dynamo_llm::tokens::{TokenBlock, TokenBlockSequence};
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

pub struct KvbmEmulationHarness {
    scheduler_tx: mpsc::UnboundedSender<SchedulerMessage>,
    transfer_client: TransferSchedulerClient,
    scheduler_task: Option<JoinHandle<anyhow::Result<()>>>,
}

impl KvbmEmulationHarness {
    pub async fn new(worker_id: &str) -> Self {
        let cancel_token = CancellationToken::new();
        let (mut scheduler, worker_client, transfer_client) =
            Scheduler::new(worker_id.to_string(), cancel_token);
        let scheduler_tx = worker_client.get_scheduler_tx();
        let scheduler_task = tokio::spawn(async move { scheduler.run().await });

        Self {
            scheduler_tx,
            transfer_client,
            scheduler_task: Some(scheduler_task),
        }
    }

    pub fn transfer_client(&self) -> TransferSchedulerClient {
        self.transfer_client.clone()
    }

    pub async fn bind_epoch(
        &self,
        key: SlotKey,
        tp_bindings: usize,
        expected_immediate_ops: usize,
    ) -> Vec<Arc<AtomicU64>> {
        let mut counters = Vec::with_capacity(tp_bindings);
        for binding in 0..tp_bindings {
            let completed = Arc::new(AtomicU64::new(0));
            self.scheduler_tx
                .send(SchedulerMessage::CreateSlot(SchedulerCreateSlotDetails {
                    key: key.clone(),
                    worker_id: format!("worker-{binding}"),
                    completed: completed.clone(),
                    expected_immediate_ops: expected_immediate_ops as u64,
                }))
                .expect("failed to bind epoch");
            counters.push(completed);
        }
        tokio::task::yield_now().await;
        counters
    }

    pub async fn complete_immediate_many(&self, key: SlotKey, count: usize) {
        for _ in 0..count {
            let handle = self
                .transfer_client
                .clone()
                .schedule_transfer(make_immediate_request(key.clone()))
                .await
                .unwrap();
            handle.mark_complete(Ok(())).await;
        }
        tokio::task::yield_now().await;
    }

    pub fn build_offload_and_onboard_cycle(
        &self,
        key: SlotKey,
        block_count: usize,
        block_size: usize,
    ) -> (LocalOffloadRequest, RemoteTransferRequest) {
        let token_blocks = make_token_blocks(block_count, block_size);
        let block_ids: Vec<usize> = (0..block_count).collect();
        let priorities: Vec<u32> = (0..block_count).map(|i| i as u32).collect();
        let operation_id = uuid::Uuid::new_v4();
        let sequence_hashes = token_blocks.iter().map(TokenBlock::sequence_hash).collect();

        let offload = LocalOffloadRequest::new(
            key.clone(),
            block_ids.clone(),
            token_blocks.clone(),
            priorities,
            operation_id,
            block_size,
            None,
            None,
        );

        let onboard = RemoteTransferRequest::from_g4_params(
            &G4OnboardParams {
                key: key.clone(),
                request_id: key.request_id.clone(),
                sequence_hashes,
                device_block_ids: block_ids,
                operation_id,
                block_size,
                token_blocks,
            },
            None,
            None,
        );

        (offload, onboard)
    }

    pub async fn finish_epoch(&self, key: SlotKey, tp_bindings: usize) {
        for binding in 0..tp_bindings {
            self.scheduler_tx
                .send(SchedulerMessage::RequestFinished(
                    SchedulerRemoveSlotDetails {
                        key: key.clone(),
                        worker_id: format!("worker-{binding}"),
                    },
                ))
                .expect("failed to finish epoch binding");
        }
        tokio::task::yield_now().await;
    }

    pub async fn wait_for_all_bindings(
        &self,
        counters: &[Arc<AtomicU64>],
        expected: u64,
        timeout: Duration,
    ) {
        let deadline = Instant::now() + timeout;
        loop {
            if counters
                .iter()
                .all(|counter| counter.load(Ordering::Acquire) == expected)
            {
                return;
            }
            assert!(
                Instant::now() < deadline,
                "timed out waiting for binding counters"
            );
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    }

    pub async fn shutdown(&mut self) {
        let (replacement_transfer_tx, _replacement_transfer_rx) = mpsc::channel(1);
        let old_transfer_client = std::mem::replace(
            &mut self.transfer_client,
            TransferSchedulerClient::new(replacement_transfer_tx),
        );
        drop(old_transfer_client);
        let (replacement_tx, _replacement_rx) = mpsc::unbounded_channel();
        let old_tx = std::mem::replace(&mut self.scheduler_tx, replacement_tx);
        drop(old_tx);
        if let Some(task) = self.scheduler_task.take() {
            task.await.unwrap().unwrap();
        }
    }
}

pub fn make_immediate_request(key: SlotKey) -> LeaderTransferRequest {
    LeaderTransferRequest {
        key,
        uuid: uuid::Uuid::new_v4(),
        requirement: None,
        request_type: RequestType::Immediate,
        chained: false,
    }
}

fn make_token_blocks(block_count: usize, block_size: usize) -> Vec<TokenBlock> {
    let tokens: Vec<u32> = (0..(block_count * block_size) as u32).collect();
    TokenBlockSequence::new(tokens.into(), block_size as u32, None)
        .blocks()
        .to_vec()
}
