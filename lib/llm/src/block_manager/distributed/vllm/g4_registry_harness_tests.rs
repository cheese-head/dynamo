// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Tier-2 harness: **mock G4 registry** + real [`PositionalRemoteHandle`] + [`super::registry_ops::match_prefix_tp_blocking`].
//!
//! This is the synchronous lookup stack used by [`super::effect_executor::execute_effect`] for
//! [`super::slot_machine::SlotEffect::StartRemoteLookup`]. It does not stand up ZMQ workers, a
//! [`super::ConnectorSlotManager`], or full [`super::KvConnectorLeaderCore`], but it fails fast if
//! the registry/handle/consensus path regresses.

#![cfg(test)]

use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;

use crate::block_manager::block::transfer::remote::RemoteKey;
use crate::block_manager::distributed::registry::{NoMetadata, OffloadResult, PositionalKey, Registry};
use crate::block_manager::distributed::PositionalRemoteHandle;

use super::registry_ops::match_prefix_tp_blocking;

/// Registry that treats every queried key as a hit (deterministic dummy [`RemoteKey`]).
#[derive(Debug, Default)]
struct AlwaysMatchRegistry;

#[async_trait]
impl Registry<PositionalKey, RemoteKey, NoMetadata> for AlwaysMatchRegistry {
    async fn register(&self, _entries: &[(PositionalKey, RemoteKey, NoMetadata)]) -> Result<()> {
        Ok(())
    }

    async fn can_offload(&self, _keys: &[PositionalKey]) -> Result<OffloadResult<PositionalKey>> {
        Ok(OffloadResult::default())
    }

    async fn match_prefix(
        &self,
        keys: &[PositionalKey],
    ) -> Result<Vec<(PositionalKey, RemoteKey, NoMetadata)>> {
        Ok(keys
            .iter()
            .map(|&k| {
                (
                    k,
                    RemoteKey::object_from_hash("kvbm-test-bucket", k.sequence_hash),
                    NoMetadata,
                )
            })
            .collect())
    }

    async fn flush(&self) -> Result<()> {
        Ok(())
    }

    async fn remove(&self, _keys: &[PositionalKey]) -> Result<usize> {
        Ok(0)
    }

    async fn touch(&self, _keys: &[PositionalKey]) -> Result<usize> {
        Ok(0)
    }
}

#[tokio::test]
async fn g4_match_prefix_tp_blocking_world_size_1_mock_hits() {
    let handle = PositionalRemoteHandle::spawn(Arc::new(AlwaysMatchRegistry::default()));
    let hashes = vec![0x1111_u64, 0x2222, 0x3333];
    let consensus = match_prefix_tp_blocking(&handle, &hashes, 1);
    assert_eq!(
        consensus, hashes,
        "TP=1 should return registry hits in sequence-hash order"
    );
}

#[tokio::test]
async fn g4_match_prefix_tp_blocking_world_size_2_consensus() {
    let handle = PositionalRemoteHandle::spawn(Arc::new(AlwaysMatchRegistry::default()));
    let hashes = vec![0xaa_u64, 0xbb];
    let consensus = match_prefix_tp_blocking(&handle, &hashes, 2);
    assert_eq!(
        consensus, hashes,
        "all workers see the same prefix in the mock → full consensus"
    );
}
