// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// kvbm-bench -- KVBM benchmark suite.
//
// Usage:
//   kvbm-bench disk setup --remote-disk-path /mnt/kvbm_cache/bench --tp 4 --users 3
//   kvbm-bench disk read  --remote-disk-path /mnt/kvbm_cache/bench --tp 4 --users 3 --concurrent-chunks 96
//   kvbm-bench disk sweep --remote-disk-path /mnt/kvbm_cache/bench --tp 4 --isl 12000 --config sweep.yaml
//
// Build:
//   cargo build --release -p dynamo-llm --bin kvbm-bench --features block-manager-bench

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

fn build_runtime(threads: usize) -> Result<tokio::runtime::Runtime> {
    let mut rt_builder = tokio::runtime::Builder::new_multi_thread();
    rt_builder.enable_all();
    if threads > 0 {
        rt_builder.worker_threads(threads);
    }
    Ok(rt_builder.build()?)
}

fn run_disk(disk: &cli::DiskArgs) -> Result<()> {
    apply_ucx_env_defaults(disk.ucx_raw_env);

    match &disk.command {
        cli::DiskCommand::Sweep { users, config, csv, .. } => {
            sysinfo::print_system_info(disk.bench_dir());
            sweep::cmd_sweep(disk, *users, config, csv)
        }
        _ => {
            let rt = build_runtime(disk.runtime_threads)?;
            let threads = disk.runtime_threads;
            eprintln!(
                "Tokio runtime: {} worker threads",
                if threads == 0 {
                    format!("{} (default)", std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1))
                } else {
                    threads.to_string()
                },
            );

            rt.block_on(async {
                match &disk.command {
                    cli::DiskCommand::Setup { users, .. } => {
                        commands::cmd_setup(disk, *users).await
                    }
                    cli::DiskCommand::Read {
                        users,
                        concurrent_chunks,
                        agent_per_chunk,
                        ..
                    } => {
                        commands::cmd_read(disk, *users, *concurrent_chunks, *agent_per_chunk).await
                    }
                    cli::DiskCommand::Sweep { .. } => unreachable!(),
                }
            })
        }
    }
}

fn main() -> Result<()> {
    if std::env::var("RUST_LOG").is_err() {
        unsafe { std::env::set_var("RUST_LOG", "error") };
    }
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let top = cli::Cli::parse();

    match &top.command {
        cli::TopCommand::Disk(disk) => run_disk(disk),
    }
}
