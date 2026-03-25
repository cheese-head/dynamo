// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::Duration;

use dynamo_runtime::config::environment_names::kvbm::remote_storage as env_g4;
use once_cell::sync::Lazy;

const DEFAULT_G4_TRANSFER_TIMEOUT_SECS: u64 = 30;
const DEFAULT_G4_MIN_CANDIDATE_BLOCKS: usize = 8;
const DEFAULT_FLUSH_BATCH_SIZE: usize = 512;

static G4_TRANSFER_TIMEOUT_SECS: AtomicU64 = AtomicU64::new(DEFAULT_G4_TRANSFER_TIMEOUT_SECS);

static G4_MIN_CANDIDATE_BLOCKS: Lazy<usize> = Lazy::new(|| {
    std::env::var(env_g4::DYN_KVBM_G4_MIN_CANDIDATE_BLOCKS)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_G4_MIN_CANDIDATE_BLOCKS)
});

static FLUSH_BATCH_SIZE: AtomicUsize = AtomicUsize::new(DEFAULT_FLUSH_BATCH_SIZE);

static INIT: Lazy<()> = Lazy::new(|| {
    if let Some(v) = std::env::var(env_g4::DYN_KVBM_G4_TRANSFER_TIMEOUT_SECS)
        .ok()
        .and_then(|s| s.parse().ok())
    {
        G4_TRANSFER_TIMEOUT_SECS.store(v, Ordering::Relaxed);
    }
    if let Some(v) = std::env::var("DYN_KVBM_FLUSH_BATCH_SIZE")
        .ok()
        .and_then(|s| s.parse().ok())
    {
        FLUSH_BATCH_SIZE.store(v, Ordering::Relaxed);
    }
});

fn ensure_init() {
    Lazy::force(&INIT);
}

#[inline]
pub fn g4_transfer_timeout() -> Duration {
    ensure_init();
    Duration::from_secs(G4_TRANSFER_TIMEOUT_SECS.load(Ordering::Relaxed))
}

pub fn set_g4_transfer_timeout_secs(secs: u64) {
    G4_TRANSFER_TIMEOUT_SECS.store(secs, Ordering::Relaxed);
}

#[inline]
pub fn g4_min_candidate_blocks() -> usize {
    *G4_MIN_CANDIDATE_BLOCKS
}

#[inline]
pub fn flush_batch_size() -> usize {
    ensure_init();
    FLUSH_BATCH_SIZE.load(Ordering::Relaxed)
}

pub fn set_flush_batch_size(size: usize) {
    FLUSH_BATCH_SIZE.store(size, Ordering::Relaxed);
}
