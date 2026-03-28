// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use clap::{Parser, Subcommand};

use dynamo_llm::block_manager::config::{
    DiskTransferFlags, DISK_FLAGS_GDS_BOTH, DISK_FLAGS_GDS_READS_ONLY, DISK_FLAGS_POSIX_BOTH,
};

#[derive(Parser)]
#[command(name = "bench_disk")]
#[command(about = "Benchmark KVBM's NIXL POSIX disk I/O path")]
pub struct Cli {
    #[arg(long, default_value_t = 4, global = true)]
    pub tp: usize,

    #[arg(long, global = true)]
    pub isl: Option<usize>,

    #[arg(long, default_value_t = 469, global = true)]
    pub num_blocks: usize,

    #[arg(long, default_value_t = 16, global = true)]
    pub chunk_size: usize,

    #[arg(long, default_value_t = 3, global = true)]
    pub iterations: usize,

    /// Enable O_DIRECT (pass --o-direct false to disable)
    #[arg(long, default_value_t = true, global = true, action = clap::ArgAction::Set)]
    pub o_direct: bool,

    /// POSIX I/O API: auto, aio, uring
    #[arg(long, default_value = "auto", global = true)]
    pub io_api: String,

    /// Disk backend: posix (default), gds (GDS_MT for both read+write),
    /// gds-read-only (POSIX write + GDS_MT read)
    #[arg(long, default_value = "posix", global = true)]
    pub disk_backend: String,

    #[arg(long, global = true)]
    pub model: Option<String>,

    #[arg(long, global = true)]
    pub block_bytes: Option<usize>,

    #[arg(long, default_value_t = 18, global = true)]
    pub num_layers: usize,

    #[arg(long, default_value_t = 2, global = true)]
    pub outer_dim: usize,

    #[arg(long, default_value_t = 256, global = true)]
    pub page_size: usize,

    #[arg(long, default_value_t = 256, global = true)]
    pub inner_dim: usize,

    #[arg(long, default_value_t = 2, global = true)]
    pub dtype_bytes: usize,

    #[arg(long, default_value_t = 0, global = true)]
    pub runtime_threads: usize,

    #[arg(long, default_value_t = false, global = true)]
    pub ucx_raw_env: bool,

    /// Print elapsed time to stderr every N seconds during I/O (0 = disable).
    #[arg(long, default_value_t = 0, global = true)]
    pub progress_interval_sec: u64,

    #[command(subcommand)]
    pub command: Command,
}

impl Cli {
    pub fn bench_dir(&self) -> &str {
        match &self.command {
            Command::Setup { dir, .. }
            | Command::Read { dir, .. }
            | Command::Sweep { dir, .. } => dir.as_str(),
        }
    }

    pub fn disk_transfer_flags(&self) -> DiskTransferFlags {
        match self.disk_backend.as_str() {
            "gds" => DISK_FLAGS_GDS_BOTH,
            "gds-read-only" | "gds-read" => DISK_FLAGS_GDS_READS_ONLY,
            _ => DISK_FLAGS_POSIX_BOTH,
        }
    }

    pub fn use_gds(&self) -> bool {
        self.disk_transfer_flags() != DISK_FLAGS_POSIX_BOTH
    }
}

#[derive(Subcommand)]
pub enum Command {
    /// Create test files (offload random data to disk via NIXL)
    Setup {
        #[arg(long)]
        dir: String,
        #[arg(long, default_value_t = 1)]
        users: usize,
    },

    /// Benchmark reading files (onboard from disk via NIXL)
    Read {
        #[arg(long)]
        dir: String,
        #[arg(long, default_value_t = 1)]
        users: usize,
        #[arg(long, default_value_t = 0)]
        concurrent_chunks: usize,
        #[arg(long, default_value_t = false)]
        agent_per_chunk: bool,
    },

    /// Sweep I/O parameters and print a comparison table
    Sweep {
        #[arg(long)]
        dir: String,
        #[arg(long, default_value_t = 3)]
        users: usize,
        /// YAML config file for sweep parameters
        #[arg(long)]
        config: Option<String>,
        /// Write CSV results to this path
        #[arg(long)]
        csv: Option<String>,
    },
}
