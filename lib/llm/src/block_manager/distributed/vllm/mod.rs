// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! vLLM integration for distributed block management.
//!
//! Provides:
//! - TP-aware registry operations (consensus lookup, multi-rank registration)
//! - Offload flow orchestration (D2H, H2O)
//! - Transfer pipeline construction

mod checksum;
mod effect_executor;
mod g4_onboard;
mod integration;
mod leader_core;
mod leader_utils;
mod offload_planner;
mod registry_ops;
mod slot_api;
mod slot_config;
mod slot_machine;
mod slot_ops;
mod slot_runtime;
mod slot_support;
mod transfer_engine;
mod transfer_signal;
mod transfer_types;

pub use effect_executor::*;
pub use integration::*;
pub use leader_core::*;
pub use leader_utils::*;
pub use slot_api::*;
pub use slot_config::*;
pub use slot_machine::*;
pub use slot_ops::*;
pub use slot_runtime::*;
pub use slot_support::*;
pub use transfer_engine::*;
pub use transfer_signal::*;
pub use transfer_types::*;

#[cfg(test)]
mod slot_phase_tests;

#[cfg(test)]
mod slot_machine_tests;

#[cfg(test)]
mod vllm_kv_semantics_tests;

#[cfg(test)]
mod kvbm_vllm_hazard_tests;

#[cfg(test)]
mod workflow_scenario_tests;

#[cfg(test)]
mod g4_registry_harness_tests;

#[cfg(test)]
mod worker_metadata_merge;

#[cfg(test)]
mod test_harness;

#[cfg(all(test, feature = "testing-cuda"))]
mod vllm_connector_e2e_tests;

#[cfg(all(test, feature = "testing-nixl", feature = "testing-cuda"))]
mod transfer_tests;

#[cfg(all(test, feature = "testing-nixl", feature = "testing-cuda"))]
mod e2e_tests;

use std::sync::Mutex;
use tokio_util::sync::CancellationToken;

static REMOTE_ABORT_TOKEN: Mutex<Option<CancellationToken>> = Mutex::new(None);

/// Get (or create) the shared remote-transfer abort token.
/// Workers use this as a child token so `clear_pool` can cancel in-flight NIXL transfers.
pub fn remote_abort_token() -> CancellationToken {
    let mut guard = REMOTE_ABORT_TOKEN.lock().unwrap();
    if let Some(ref token) = *guard {
        token.clone()
    } else {
        let token = CancellationToken::new();
        *guard = Some(token.clone());
        token
    }
}

/// Cancel the current abort token and replace it with a fresh one.
/// Called by `clear_pool` to abort in-flight NIXL transfers.
pub fn cancel_remote_transfers() {
    let mut guard = REMOTE_ABORT_TOKEN.lock().unwrap();
    if let Some(old) = guard.take() {
        old.cancel();
        tracing::info!("Cancelled remote abort token to abort in-flight NIXL transfers");
    }
    *guard = Some(CancellationToken::new());
}
