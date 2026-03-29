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
use dynamo_llm::block_manager::{PinnedStorage, SystemStorage};

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

    let use_system = disk.host_storage.eq_ignore_ascii_case("system");

    match &disk.command {
        cli::DiskCommand::Sweep { users, config, csv, .. } => {
            sysinfo::print_system_info(disk.bench_dir());
            if use_system {
                sweep::cmd_sweep::<SystemStorage>(disk, *users, config, csv)
            } else {
                sweep::cmd_sweep::<PinnedStorage>(disk, *users, config, csv)
            }
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
                        if use_system {
                            commands::cmd_setup::<SystemStorage>(disk, *users).await
                        } else {
                            commands::cmd_setup::<PinnedStorage>(disk, *users).await
                        }
                    }
                    cli::DiskCommand::Read {
                        users,
                        concurrent_chunks,
                        agent_per_chunk,
                        agent_pool_size,
                        ..
                    } => {
                        if use_system {
                            commands::cmd_read::<SystemStorage>(disk, *users, *concurrent_chunks, *agent_per_chunk, *agent_pool_size).await
                        } else {
                            commands::cmd_read::<PinnedStorage>(disk, *users, *concurrent_chunks, *agent_per_chunk, *agent_pool_size).await
                        }
                    }
                    cli::DiskCommand::Sweep { .. } => unreachable!(),
                }
            })
        }
    }
}

fn install_signal_handler() {
    unsafe {
        libc::signal(libc::SIGINT, handle_sigint as libc::sighandler_t);
        libc::signal(libc::SIGTERM, handle_sigint as libc::sighandler_t);
    }
}

extern "C" fn handle_sigint(_sig: libc::c_int) {
    let msg = b"\nInterrupted -- exiting.\n";
    unsafe { libc::write(2, msg.as_ptr() as *const libc::c_void, msg.len()) };
    std::process::exit(130);
}

fn main() -> Result<()> {
    install_signal_handler();

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
