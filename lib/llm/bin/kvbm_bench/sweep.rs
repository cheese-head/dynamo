// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Result;
use serde::Deserialize;
use tokio_util::sync::CancellationToken;

use dynamo_llm::block_manager::block::transfer::{clear_remote_disk_fd_cache, take_fd_open_durations};
use dynamo_llm::block_manager::block::transfer::remote::RemoteTransferPipeline;

use crate::cli::DiskArgs;
use crate::layout::{effective_num_blocks, resolve_layout};
use crate::table;
use crate::worker::{self, Worker};

/// YAML sweep configuration.  If `configs` is present, each entry is run as-is.
/// Otherwise, the top-level arrays generate a cartesian product.
#[derive(Deserialize, Debug, Clone)]
#[serde(untagged)]
pub enum SweepFile {
    Explicit {
        configs: Vec<SweepPoint>,
    },
    Product {
        #[serde(default = "default_io_api")]
        nixl_posix_api: Vec<String>,
        #[serde(default = "default_o_direct")]
        remote_disk_o_direct: Vec<bool>,
        #[serde(default = "default_disk_backend")]
        remote_disk_backend: Vec<String>,
        #[serde(default = "default_chunk_size")]
        chunk_size: Vec<usize>,
        #[serde(default = "default_concurrent_chunks")]
        concurrent_chunks: Vec<usize>,
        #[serde(default = "default_agent_per_chunk")]
        agent_per_chunk: Vec<bool>,
        #[serde(default = "default_runtime_threads")]
        runtime_threads: Vec<usize>,
    },
}

fn default_io_api() -> Vec<String> { vec!["auto".into()] }
fn default_o_direct() -> Vec<bool> { vec![true, false] }
fn default_disk_backend() -> Vec<String> { vec!["posix".into()] }
fn default_chunk_size() -> Vec<usize> { vec![16] }
fn default_concurrent_chunks() -> Vec<usize> { vec![0] }
fn default_agent_per_chunk() -> Vec<bool> { vec![false] }
fn default_runtime_threads() -> Vec<usize> { vec![0] }

#[derive(Deserialize, Debug, Clone)]
pub struct SweepPoint {
    #[serde(default = "default_io_api_single")]
    pub nixl_posix_api: String,
    #[serde(default = "default_o_direct_single")]
    pub remote_disk_o_direct: bool,
    #[serde(default = "default_disk_backend_single")]
    pub remote_disk_backend: String,
    #[serde(default = "default_chunk_size_single")]
    pub chunk_size: usize,
    #[serde(default)]
    pub concurrent_chunks: usize,
    #[serde(default)]
    pub agent_per_chunk: bool,
    #[serde(default)]
    pub runtime_threads: usize,
}

fn default_io_api_single() -> String { "auto".into() }
fn default_o_direct_single() -> bool { true }
fn default_disk_backend_single() -> String { "posix".into() }
fn default_chunk_size_single() -> usize { 16 }

impl SweepFile {
    pub fn into_points(self) -> Vec<SweepPoint> {
        match self {
            SweepFile::Explicit { configs } => configs,
            SweepFile::Product {
                nixl_posix_api, remote_disk_o_direct, remote_disk_backend,
                chunk_size, concurrent_chunks, agent_per_chunk,
                runtime_threads,
            } => {
                let mut points = Vec::new();
                for api in &nixl_posix_api {
                    for &od in &remote_disk_o_direct {
                        for be in &remote_disk_backend {
                            for &cs in &chunk_size {
                                for &cc in &concurrent_chunks {
                                    for &apc in &agent_per_chunk {
                                        for &rt in &runtime_threads {
                                            points.push(SweepPoint {
                                                nixl_posix_api: api.clone(),
                                                remote_disk_o_direct: od,
                                                remote_disk_backend: be.clone(),
                                                chunk_size: cs,
                                                concurrent_chunks: cc,
                                                agent_per_chunk: apc,
                                                runtime_threads: rt,
                                            });
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                points
            }
        }
    }

    pub fn default_sweep() -> Self {
        SweepFile::Product {
            nixl_posix_api: vec!["auto".into(), "uring".into()],
            remote_disk_o_direct: vec![true, false],
            remote_disk_backend: vec!["posix".into()],
            chunk_size: vec![16, 64],
            concurrent_chunks: vec![0, 64],
            agent_per_chunk: vec![false],
            runtime_threads: vec![0],
        }
    }
}

pub struct SweepResult {
    pub point: SweepPoint,
    pub total_bytes: usize,
    pub read_durations: Vec<Duration>,
    pub write_duration: Option<Duration>,
    pub fd_open_p99: Option<Duration>,
    pub error: Option<String>,
}

pub struct DurationStats {
    pub min: Duration,
    pub max: Duration,
    pub avg: Duration,
    pub p50: Duration,
    pub p95: Duration,
    pub p99: Duration,
}

impl DurationStats {
    pub fn from_durations(durations: &[Duration]) -> Option<Self> {
        if durations.is_empty() {
            return None;
        }
        let mut sorted: Vec<Duration> = durations.to_vec();
        sorted.sort();
        let n = sorted.len();
        let avg = sorted.iter().sum::<Duration>() / n as u32;
        Some(Self {
            min: sorted[0],
            max: sorted[n - 1],
            avg,
            p50: sorted[n * 50 / 100],
            p95: sorted[(n * 95 / 100).min(n - 1)],
            p99: sorted[(n * 99 / 100).min(n - 1)],
        })
    }

    pub fn gbps(&self, total_bytes: usize) -> f64 {
        total_bytes as f64 / self.avg.as_secs_f64() / 1e9
    }

    pub fn best_gbps(&self, total_bytes: usize) -> f64 {
        total_bytes as f64 / self.min.as_secs_f64() / 1e9
    }
}

impl SweepResult {
    pub fn read_stats(&self) -> Option<DurationStats> {
        DurationStats::from_durations(&self.read_durations)
    }

    pub fn avg_gbps(&self) -> f64 {
        self.read_stats().map(|s| s.gbps(self.total_bytes)).unwrap_or(0.0)
    }

    pub fn best_gbps(&self) -> f64 {
        self.read_stats().map(|s| s.best_gbps(self.total_bytes)).unwrap_or(0.0)
    }
}

pub fn cmd_sweep(
    cli: &DiskArgs,
    num_users: usize,
    config_path: &Option<String>,
    csv_path: &Option<String>,
) -> Result<()> {
    let sweep_file = if let Some(path) = config_path {
        let contents = std::fs::read_to_string(path)?;
        serde_yaml::from_str(&contents)?
    } else {
        SweepFile::default_sweep()
    };

    let points = sweep_file.into_points();
    let total_configs = points.len();

    let resolved = resolve_layout(cli);
    let bb = resolved.block_bytes();
    let nb = effective_num_blocks(cli);
    let per_user_per_worker = nb * bb;
    let total_bytes = per_user_per_worker * cli.tp * num_users;

    println!("Sweep: {total_configs} configurations to test");
    println!("  {:.2} GB per point ({} workers × {} users × {} blocks × {} B)",
        total_bytes as f64 / 1e9, cli.tp, num_users, nb, bb);
    println!();

    let mut results: Vec<SweepResult> = Vec::new();

    for (idx, point) in points.into_iter().enumerate() {
        let rt_threads = if point.runtime_threads > 0 {
            point.runtime_threads
        } else if cli.runtime_threads > 0 {
            cli.runtime_threads
        } else {
            0
        };

        let mut rt_builder = tokio::runtime::Builder::new_multi_thread();
        rt_builder.enable_all();
        if rt_threads > 0 {
            rt_builder.worker_threads(rt_threads);
        }
        let rt = rt_builder.build()?;

        let point_dir = format!("{}/point_{idx}", cli.bench_dir());
        let threads_label = if rt_threads == 0 { "default".to_string() } else { rt_threads.to_string() };
        let label = format!(
            "[{}/{}] io_api={} o_direct={} backend={} chunk_size={} conc_chunks={} rt_threads={}",
            idx + 1, total_configs,
            point.nixl_posix_api, point.remote_disk_o_direct, point.remote_disk_backend,
            point.chunk_size, point.concurrent_chunks, threads_label,
        );
        eprintln!("{label}");

        unsafe {
            std::env::set_var(
                "DYN_KVBM_REMOTE_DISK_O_DIRECT",
                if point.remote_disk_o_direct { "true" } else { "false" },
            );
        }

        rt.block_on(clear_remote_disk_fd_cache());

        let use_gds = point.remote_disk_backend != "posix";
        let disk_flags = match point.remote_disk_backend.as_str() {
            "gds" => dynamo_llm::block_manager::config::DISK_FLAGS_GDS_BOTH,
            "gds-read-only" | "gds-read" => dynamo_llm::block_manager::config::DISK_FLAGS_GDS_READS_ONLY,
            _ => dynamo_llm::block_manager::config::DISK_FLAGS_POSIX_BOTH,
        };

        let chunk_sz = if point.chunk_size == 0 { nb } else { point.chunk_size };

        let agent_result = std::panic::catch_unwind(|| {
            worker::build_agent("sweep-probe", &point.nixl_posix_api, use_gds, cli.gds_threads)
        });

        if agent_result.is_err() {
            eprintln!("  SKIPPED (backend unavailable)");
            results.push(SweepResult {
                point,
                total_bytes,
                read_durations: vec![],
                write_duration: None,
                fd_open_p99: None,
                error: Some("backend unavailable".into()),
            });
            continue;
        }
        drop(agent_result);

        // --- Write phase ---
        std::fs::create_dir_all(&point_dir)?;
        let layout_cfg = worker::build_layout_config(cli, &resolved);
        let write_start = Instant::now();
        let setup_err = rt.block_on(async {
            let mut handles = Vec::new();
            for wid in 0..cli.tp {
                for uid in 0..num_users {
                    let dir = point_dir.clone();
                    let io_api = point.nixl_posix_api.clone();
                    let o_direct = point.remote_disk_o_direct;
                    let gds_threads = cli.gds_threads;
                    let tp = cli.tp;
                    let layout_cfg = layout_cfg.clone();

                    handles.push(tokio::spawn(async move {
                        unsafe { std::env::set_var("DYN_KVBM_REMOTE_DISK_O_DIRECT", if o_direct { "true" } else { "false" }) };
                        let agent = worker::build_agent(&format!("setup-w{wid}-u{uid}"), &io_api, use_gds, gds_threads);
                        let (_layout, blocks) = worker::allocate_and_register(layout_cfg, &agent);
                        let nb = blocks.len();
                        let descs = worker::make_descriptors(&dir, nb, bb, wid, tp, uid);
                        let remote_ctx = worker::build_remote_ctx(Arc::new(Some(agent)), &dir, wid, tp, disk_flags);
                        let cancel = CancellationToken::new();
                        let pipeline = RemoteTransferPipeline::offload_direct(descs);
                        pipeline.execute(&blocks, remote_ctx.as_ref(), &cancel).await
                            .map_err(|e| anyhow::anyhow!("setup worker {wid} user {uid}: {e}"))
                    }));
                }
            }
            let r: Result<()> = async {
                for h in handles { h.await??; }
                Ok(())
            }.await;
            r
        });
        if let Err(e) = setup_err {
            eprintln!("  SETUP ERROR: {e}");
            results.push(SweepResult {
                point,
                total_bytes,
                read_durations: vec![],
                write_duration: None,
                    fd_open_p99: None,
                    error: Some(format!("setup: {e}")),
            });
            continue;
        }
        let write_duration = write_start.elapsed();
        rt.block_on(clear_remote_disk_fd_cache());
        let _ = take_fd_open_durations();
        let write_gbps = total_bytes as f64 / write_duration.as_secs_f64() / 1e9;
        eprintln!("  write: {:.3}s ({write_gbps:.2} GB/s) → {point_dir}", write_duration.as_secs_f64());

        // --- Read phase ---
        let (durations, had_error) = rt.block_on(async {
            let workers: Vec<Worker> = (0..cli.tp)
                .map(|wid| {
                    let agent = worker::build_agent(
                        &format!("sweep-w{wid}"),
                        &point.nixl_posix_api,
                        use_gds,
                        cli.gds_threads,
                    );
                    let mut user_layouts = Vec::with_capacity(num_users);
                    let mut user_blocks = Vec::with_capacity(num_users);
                    let mut user_descriptors = Vec::with_capacity(num_users);

                    for uid in 0..num_users {
                        let config = worker::build_layout_config(cli, &resolved);
                        let (layout, blocks) = worker::allocate_and_register(config, &agent);
                        let descs = worker::make_descriptors(&point_dir, nb, bb, wid, cli.tp, uid);
                        user_layouts.push(layout);
                        user_blocks.push(blocks);
                        user_descriptors.push(descs);
                    }

                    let remote_ctx = worker::build_remote_ctx(
                        std::sync::Arc::new(Some(agent.clone())),
                        &point_dir,
                        wid,
                        cli.tp,
                        disk_flags,
                    );

                    Worker {
                        id: wid,
                        _agent: agent,
                        user_layouts,
                        user_blocks,
                        user_descriptors,
                        remote_ctx,
                    }
                })
                .collect();

            let mut durations = Vec::new();
            let mut had_error = None;

            for iter in 0..cli.iterations {
                let cancel = CancellationToken::new();
                let start = Instant::now();

                let mut handles = Vec::new();
                for w in &workers {
                    for uid in 0..num_users {
                        let blocks = w.user_blocks[uid].clone();
                        let descs = w.user_descriptors[uid].clone();
                        let ctx = w.remote_ctx.clone();
                        let cancel = cancel.clone();
                        let io_api = point.nixl_posix_api.clone();
                        let use_gds = use_gds;
                        let gds_threads = cli.gds_threads;
                        let disk_flags = disk_flags;
                        let wid = w.id;
                        let concurrent_chunks = point.concurrent_chunks;
                        let agent_per_chunk = point.agent_per_chunk;

                        handles.push(tokio::spawn(async move {
                            worker::run_chunked_pipeline(
                                &blocks, &descs, &ctx,
                                chunk_sz, concurrent_chunks, agent_per_chunk,
                                &io_api, use_gds, gds_threads, disk_flags, wid, &cancel,
                            ).await
                        }));
                    }
                }

                let read_result: Result<()> = async {
                    for h in handles {
                        h.await??;
                    }
                    Ok(())
                }
                .await;

                match read_result {
                    Ok(()) => {
                        let elapsed = start.elapsed();
                        let tp = total_bytes as f64 / elapsed.as_secs_f64() / 1e9;
                        eprintln!("  iter {iter}: {:.3}s ({tp:.2} GB/s)", elapsed.as_secs_f64());
                        durations.push(elapsed);
                    }
                    Err(e) => {
                        eprintln!("  iter {iter}: ERROR {e}");
                        had_error = Some(format!("{e}"));
                        break;
                    }
                }
            }

            // Explicitly drop workers (NIXL agents, registered memory, remote contexts)
            // before the async block returns, while the Tokio runtime is still alive.
            drop(workers);
            clear_remote_disk_fd_cache().await;

            (durations, had_error)
        });

        let mut fd_opens = take_fd_open_durations();
        fd_opens.sort();
        let fd_p99 = if fd_opens.is_empty() {
            None
        } else {
            Some(fd_opens[(fd_opens.len() * 99 / 100).min(fd_opens.len() - 1)])
        };

        results.push(SweepResult {
            point,
            total_bytes,
            read_durations: durations,
            write_duration: Some(write_duration),
            fd_open_p99: fd_p99,
            error: had_error,
        });
    }

    results.sort_by(|a, b| b.best_gbps().partial_cmp(&a.best_gbps()).unwrap());

    println!();
    table::print_sweep_table(&results);

    if let Some(path) = csv_path {
        table::write_csv(&results, path)?;
        println!("\nResults written to {path}");
    }

    Ok(())
}
