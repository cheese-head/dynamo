// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::block_manager::connector::protocol::{RequestType, TransferType, WorkerTransferRequest};
use crate::tokens::TokenBlock;

/// Parameters for G4 onboard operation.
#[derive(Debug, Clone)]
pub struct G4OnboardParams {
    pub request_id: String,
    pub sequence_hashes: Vec<u64>,
    pub device_block_ids: Vec<usize>,
    pub operation_id: uuid::Uuid,
    pub block_size: usize,
    pub token_blocks: Vec<TokenBlock>,
}

/// Prepare G4 onboard operation.
pub fn onboard_from_g4(
    request_id: String,
    sequence_hashes: Vec<u64>,
    device_block_ids: Vec<usize>,
    block_size: usize,
    token_blocks: Vec<TokenBlock>,
) -> (G4OnboardParams, Vec<WorkerTransferRequest>) {
    let num_blocks = sequence_hashes.len();
    debug_assert!(device_block_ids.is_empty() || device_block_ids.len() == num_blocks);
    debug_assert!(token_blocks.is_empty() || token_blocks.len() == num_blocks);
    let operation_id = uuid::Uuid::new_v4();

    tracing::debug!(
        target: "kvbm-g4",
        request_id = %request_id,
        operation_id = %operation_id,
        num_blocks = num_blocks,
        "preparing onboard for {} blocks",
        num_blocks
    );

    let params = G4OnboardParams {
        request_id: request_id.clone(),
        sequence_hashes,
        device_block_ids: device_block_ids.clone(),
        operation_id,
        block_size,
        token_blocks,
    };

    let worker_reqs = if device_block_ids.is_empty() {
        Vec::new()
    } else {
        vec![WorkerTransferRequest {
            request_id: request_id.clone(),
            uuid: operation_id,
            transfer_type: TransferType::Load,
            request_type: RequestType::Immediate,
            block_ids: device_block_ids,
        }]
    };

    (params, worker_reqs)
}
