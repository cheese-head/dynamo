// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use clap::{Parser, Subcommand, Args};

use dynamo_llm::block_manager::config::{
    DiskTransferFlags, DISK_FLAGS_GDS_BOTH, DISK_FLAGS_GDS_READS_ONLY, DISK_FLAGS_POSIX_BOTH,
};

#[derive(Parser)]
#[command(name = "kvbm-bench")]
#[command(about = "KVBM benchmark suite")]
pub struct Cli {
    #[command(subcommand)]
    pub command: TopCommand,
}

#[derive(Subcommand)]
pub enum TopCommand {
    /// Benchmark KVBM's NIXL disk I/O path
    Disk(DiskArgs),
}

#[derive(Args)]
pub struct DiskArgs {
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

    /// Enable O_DIRECT (env: DYN_KVBM_REMOTE_DISK_O_DIRECT)
    #[arg(long = "remote-disk-o-direct", default_value_t = true, global = true, action = clap::ArgAction::Set)]
    pub o_direct: bool,

    /// POSIX I/O API: auto, aio, uring (env: DYN_KVBM_NIXL_POSIX_API)
    #[arg(long = "nixl-posix-api", default_value = "auto", global = true)]
    pub io_api: String,

    /// Disk backend: posix (default), gds (GDS_MT for both read+write),
    /// gds-read-only (POSIX write + GDS_MT read)
    /// (env: DYN_KVBM_REMOTE_DISK_USE_GDS + DYN_KVBM_REMOTE_DISK_GDS_READS_ONLY)
    #[arg(long = "remote-disk-backend", default_value = "posix", global = true)]
    pub disk_backend: String,

    /// GDS_MT thread count (0 = NIXL default, which is hardware_concurrency / 2)
    #[arg(long = "gds-threads", default_value_t = 0, global = true)]
    pub gds_threads: usize,

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

    /// Host memory allocator: pinned (CUDA page-locked, default) or system (malloc).
    /// System storage with O_DIRECT requires DYN_KVBM_BOUNCE_BUFFER=1 for alignment.
    #[arg(long = "host-storage", default_value = "pinned", global = true)]
    pub host_storage: String,

    /// NIXL async completion poll interval in microseconds (sets `DYN_KVBM_NIXL_POLL_INTERVAL_US`).
    /// 0 = unset (library default 50_000 µs). Sweep YAML `nixl_poll_interval_us` overrides this per point when non-zero.
    #[arg(long = "nixl-poll-interval-us", default_value_t = 0, global = true)]
    pub nixl_poll_interval_us: u64,

    #[command(subcommand)]
    pub command: DiskCommand,
}

impl DiskArgs {
    pub fn bench_dir(&self) -> &str {
        match &self.command {
            DiskCommand::Setup { dir, .. }
            | DiskCommand::Read { dir, .. }
            | DiskCommand::Sweep { dir, .. } => dir.as_str(),
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
pub enum DiskCommand {
    /// Create test files (offload random data to disk via NIXL)
    Setup {
        /// Benchmark directory (env: DYN_KVBM_REMOTE_DISK_PATHS)
        #[arg(long = "remote-disk-path")]
        dir: String,
        #[arg(long, default_value_t = 1)]
        users: usize,
    },

    /// Benchmark reading files (onboard from disk via NIXL)
    Read {
        /// Benchmark directory (env: DYN_KVBM_REMOTE_DISK_PATHS)
        #[arg(long = "remote-disk-path")]
        dir: String,
        #[arg(long, default_value_t = 1)]
        users: usize,
        #[arg(long, default_value_t = 0)]
        concurrent_chunks: usize,
        #[arg(long, default_value_t = false)]
        agent_per_chunk: bool,
        #[arg(long, default_value_t = 0)]
        agent_pool_size: usize,
    },

    /// Sweep I/O parameters and print a comparison table
    Sweep {
        /// Benchmark directory (env: DYN_KVBM_REMOTE_DISK_PATHS)
        #[arg(long = "remote-disk-path")]
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
