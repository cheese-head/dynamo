// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// bench_disk -- Benchmark KVBM's NIXL POSIX disk I/O path.
//
// Usage:
//   bench_disk setup --dir /mnt/kv_cache/bench --tp 4 --users 3
//   bench_disk read  --dir /mnt/kv_cache/bench --tp 4 --users 3 --concurrent-chunks 96
//   bench_disk sweep --dir /mnt/kv_cache/bench --tp 4 --isl 12000 --config sweep.yaml
//
// Build:
//   cargo build --release -p dynamo-llm --bin bench_disk --features block-manager-bench

mod cli;
mod commands;
mod layout;
mod sweep;
mod sysinfo;
mod table;
mod worker;

use anyhow::Result;
use clap::Parser;

fn apply_ucx_env_defaults(ucx_raw_env: bool) {
    if ucx_raw_env || std::env::var_os("UCX_TLS").is_some() {
        return;
    }
    const DEFAULT_TLS: &str = "sm,self,cuda_copy,cuda_ipc,tcp";
    unsafe {
        std::env::set_var("UCX_TLS", DEFAULT_TLS);
    }
    tracing::info!(
        ucx_tls = DEFAULT_TLS,
        "UCX_TLS was unset; using local-friendly defaults (use --ucx-raw-env or set UCX_TLS to override)"
    );
}

fn main() -> Result<()> {
    if std::env::var("RUST_LOG").is_err() {
        unsafe { std::env::set_var("RUST_LOG", "warn,bench_disk=info") };
    }
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let cli = cli::Cli::parse();
    apply_ucx_env_defaults(cli.ucx_raw_env);

    let mut rt_builder = tokio::runtime::Builder::new_multi_thread();
    rt_builder.enable_all();
    if cli.runtime_threads > 0 {
        rt_builder.worker_threads(cli.runtime_threads);
    }
    let rt = rt_builder.build()?;

    let threads = cli.runtime_threads;
    eprintln!(
        "Tokio runtime: {} worker threads",
        if threads == 0 {
            format!("{} (default)", std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1))
        } else {
            threads.to_string()
        },
    );

    rt.block_on(async {
        match &cli.command {
            cli::Command::Setup { users, .. } => {
                commands::cmd_setup(&cli, *users).await
            }
            cli::Command::Read {
                users,
                concurrent_chunks,
                agent_per_chunk,
                ..
            } => {
                commands::cmd_read(&cli, *users, *concurrent_chunks, *agent_per_chunk).await
            }
            cli::Command::Sweep { users, config, csv, .. } => {
                sysinfo::print_system_info(cli.bench_dir());
                sweep::cmd_sweep(&cli, *users, config, csv).await
            }
        }
    })
}
