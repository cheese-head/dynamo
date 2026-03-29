// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;
use std::time::Instant;

use anyhow::Result;
use tokio_util::sync::CancellationToken;

use dynamo_llm::block_manager::block::transfer::{
    remote::RemoteTransferPipeline, NIXL_POLL_INTERVAL_US_ENV,
};

use crate::cli::DiskArgs;
use crate::layout::{effective_num_blocks, resolve_layout};
use crate::table;
use crate::worker::{
    BenchStorage, Worker, build_agent, build_layout_config, build_remote_ctx,
    make_descriptors, allocate_and_register, run_chunked_pipeline,
    spawn_progress_heartbeat,
};

/// Sets or clears `DYN_KVBM_NIXL_POLL_INTERVAL_US`. `point_us` wins over `cli_us` when non-zero.
pub(crate) fn apply_nixl_poll_interval_env(cli_us: u64, point_us: u64) {
    let eff = if point_us > 0 {
        point_us
    } else if cli_us > 0 {
        cli_us
    } else {
        unsafe { std::env::remove_var(NIXL_POLL_INTERVAL_US_ENV) };
        return;
    };
    unsafe {
        std::env::set_var(NIXL_POLL_INTERVAL_US_ENV, eff.to_string());
    }
}

pub async fn cmd_setup<S: BenchStorage>(cli: &DiskArgs, num_users: usize) -> Result<()> {
    let resolved = resolve_layout(cli);
    let bb = resolved.block_bytes();
    let nb = effective_num_blocks(cli);
    let per_user_per_worker = nb * bb;
    let total = per_user_per_worker * cli.tp * num_users;

    println!("Setup");
    println!("  Layout:      {resolved}");
    if let Some(isl) = cli.isl {
        println!("  ISL:         {} tokens → {} blocks", isl, nb);
    }
    println!("  Users:       {num_users}");
    println!("  Host memory: {}", S::name());
    println!("  Files:       {} workers × {} users × {} blocks × {} B = {:.2} GB → {}",
        cli.tp, num_users, nb, bb, total as f64 / 1e9, cli.bench_dir());

    std::fs::create_dir_all(cli.bench_dir())?;
    apply_nixl_poll_interval_env(cli.nixl_poll_interval_us, 0);
    unsafe { std::env::set_var("DYN_KVBM_REMOTE_DISK_O_DIRECT", if cli.o_direct { "true" } else { "false" }) };

    let layout_cfg = build_layout_config(cli, &resolved);
    let start = Instant::now();
    let progress_cancel = CancellationToken::new();
    spawn_progress_heartbeat(cli.progress_interval_sec, "setup".to_string(), start, &progress_cancel);

    let mut handles = Vec::new();
    for wid in 0..cli.tp {
        for uid in 0..num_users {
            let dir = cli.bench_dir().to_string();
            let io_api = cli.io_api.clone();
            let o_direct = cli.o_direct;
            let use_gds = cli.use_gds();
            let gds_threads = cli.gds_threads;
            let disk_flags = cli.disk_transfer_flags();
            let tp = cli.tp;
            let layout_cfg = layout_cfg.clone();

            handles.push(tokio::spawn(async move {
                unsafe { std::env::set_var("DYN_KVBM_REMOTE_DISK_O_DIRECT", if o_direct { "true" } else { "false" }) };

                let agent = build_agent(&format!("setup-w{wid}-u{uid}"), &io_api, use_gds, gds_threads);
                let (_layout, blocks) = allocate_and_register::<S>(layout_cfg, &agent);

                let nb = blocks.len();
                let descs = make_descriptors(&dir, nb, bb, wid, tp, uid);
                let remote_ctx = build_remote_ctx(Arc::new(Some(agent)), &dir, wid, tp, disk_flags);
                let cancel = CancellationToken::new();

                let pipeline = RemoteTransferPipeline::offload_direct(descs);
                pipeline.execute(&blocks, remote_ctx.as_ref(), &cancel).await
                    .map_err(|e| anyhow::anyhow!("setup worker {wid} user {uid}: {e}"))
            }));
        }
    }

    let setup_result: Result<()> = async {
        for h in handles { h.await??; }
        Ok(())
    }.await;
    progress_cancel.cancel();
    setup_result?;

    let elapsed = start.elapsed();
    println!("Setup complete: {:.3} s ({:.2} GB/s)",
        elapsed.as_secs_f64(), total as f64 / elapsed.as_secs_f64() / 1e9);
    Ok(())
}

pub async fn cmd_read<S: BenchStorage>(
    cli: &DiskArgs,
    num_users: usize,
    concurrent_chunks: usize,
    agent_per_chunk: bool,
    agent_pool_size: usize,
) -> Result<()> {
    let resolved = resolve_layout(cli);
    let bb = resolved.block_bytes();
    let nb = effective_num_blocks(cli);
    let per_user_per_worker = nb * bb;
    let total = per_user_per_worker * cli.tp * num_users;
    let chunk_sz = if cli.chunk_size == 0 { nb } else { cli.chunk_size };
    let num_chunks = (nb + chunk_sz - 1) / chunk_sz;

    println!("Read benchmark");
    println!("  Layout:      {resolved}");
    println!("  TP:          {}", cli.tp);
    println!("  Users:       {num_users} concurrent requests/worker");
    if let Some(isl) = cli.isl {
        println!("  ISL:         {} tokens → {} blocks/user", isl, nb);
    }
    println!("  Files:       {} workers × {} users × {} blocks × {} B = {:.2} GB",
        cli.tp, num_users, nb, bb, total as f64 / 1e9);
    println!("  Chunk size:  {} blocks/request ({} chunks/user)", chunk_sz, num_chunks);
    println!("  Concurrent:  {}", if concurrent_chunks == 0 { "serial".into() } else { format!("{concurrent_chunks} chunks/worker (across all users)") });
    println!("  Agent mode:  {}", if agent_per_chunk { "per-chunk" } else { "shared (KVBM default)" });
    println!("  Host memory: {}", S::name());
    println!("  O_DIRECT:    {}", cli.o_direct);
    println!("  I/O API:     {}", cli.io_api);
    println!("  Backend:     {}", cli.disk_backend);
    println!();

    apply_nixl_poll_interval_env(cli.nixl_poll_interval_us, 0);
    unsafe { std::env::set_var("DYN_KVBM_REMOTE_DISK_O_DIRECT", if cli.o_direct { "true" } else { "false" }) };

    let workers: Vec<Worker<S>> = (0..cli.tp)
        .map(|wid| Worker::new(cli, &resolved, wid, num_users))
        .collect();
    eprintln!("Allocated {} workers × {} users × {} blocks × {} B = {:.1} GB {} total",
        cli.tp, num_users, nb, bb, total as f64 / 1e9, S::name());

    let mut durations = Vec::new();

    for iter in 0..cli.iterations {
        let cancel = CancellationToken::new();
        let start = Instant::now();
        let progress_cancel = CancellationToken::new();
        spawn_progress_heartbeat(cli.progress_interval_sec, format!("read iter {iter}"), start, &progress_cancel);

        let mut handles = Vec::new();
        for w in &workers {
            for uid in 0..num_users {
                let blocks = w.user_blocks[uid].clone();
                let descs = w.user_descriptors[uid].clone();
                let ctx = w.remote_ctx.clone();
                let cancel = cancel.clone();
                let io_api = cli.io_api.clone();
                let use_gds = cli.use_gds();
                let gds_threads = cli.gds_threads;
                let disk_flags = cli.disk_transfer_flags();
                let wid = w.id;

                handles.push(tokio::spawn(async move {
                    run_chunked_pipeline::<S>(
                        &blocks, &descs, &ctx,
                        chunk_sz, concurrent_chunks, agent_per_chunk,
                        agent_pool_size,
                        &io_api, use_gds, gds_threads, disk_flags, wid, &cancel,
                    ).await
                }));
            }
        }

        let read_result: Result<()> = async {
            for h in handles { h.await??; }
            Ok(())
        }.await;
        progress_cancel.cancel();
        read_result?;

        let elapsed = start.elapsed();
        let label = if iter == 0 { " (cold FD cache)" } else { " (warm FD cache)" };
        let tp = total as f64 / elapsed.as_secs_f64() / 1e9;
        println!("  iter {iter}: {:.3} s  ({tp:.2} GB/s){label}", elapsed.as_secs_f64());
        durations.push(elapsed);
    }

    table::print_results("Read", total, &durations);
    Ok(())
}
