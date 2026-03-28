// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::time::{Duration, Instant};

use anyhow::Result;
use serde::Deserialize;
use tokio_util::sync::CancellationToken;

use crate::cli::Cli;
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
        io_api: Vec<String>,
        #[serde(default = "default_o_direct")]
        o_direct: Vec<bool>,
        #[serde(default = "default_disk_backend")]
        disk_backend: Vec<String>,
        #[serde(default = "default_chunk_size")]
        chunk_size: Vec<usize>,
        #[serde(default = "default_concurrent_chunks")]
        concurrent_chunks: Vec<usize>,
        #[serde(default = "default_agent_per_chunk")]
        agent_per_chunk: Vec<bool>,
    },
}

fn default_io_api() -> Vec<String> { vec!["auto".into()] }
fn default_o_direct() -> Vec<bool> { vec![true, false] }
fn default_disk_backend() -> Vec<String> { vec!["posix".into()] }
fn default_chunk_size() -> Vec<usize> { vec![16] }
fn default_concurrent_chunks() -> Vec<usize> { vec![0] }
fn default_agent_per_chunk() -> Vec<bool> { vec![false] }

#[derive(Deserialize, Debug, Clone)]
pub struct SweepPoint {
    #[serde(default = "default_io_api_single")]
    pub io_api: String,
    #[serde(default = "default_o_direct_single")]
    pub o_direct: bool,
    #[serde(default = "default_disk_backend_single")]
    pub disk_backend: String,
    #[serde(default = "default_chunk_size_single")]
    pub chunk_size: usize,
    #[serde(default)]
    pub concurrent_chunks: usize,
    #[serde(default)]
    pub agent_per_chunk: bool,
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
                io_api, o_direct, disk_backend,
                chunk_size, concurrent_chunks, agent_per_chunk,
            } => {
                let mut points = Vec::new();
                for api in &io_api {
                    for &od in &o_direct {
                        for be in &disk_backend {
                            for &cs in &chunk_size {
                                for &cc in &concurrent_chunks {
                                    for &apc in &agent_per_chunk {
                                        points.push(SweepPoint {
                                            io_api: api.clone(),
                                            o_direct: od,
                                            disk_backend: be.clone(),
                                            chunk_size: cs,
                                            concurrent_chunks: cc,
                                            agent_per_chunk: apc,
                                        });
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
            io_api: vec!["auto".into(), "uring".into()],
            o_direct: vec![true, false],
            disk_backend: vec!["posix".into()],
            chunk_size: vec![16, 64],
            concurrent_chunks: vec![0, 64],
            agent_per_chunk: vec![false],
        }
    }
}

pub struct SweepResult {
    pub point: SweepPoint,
    pub total_bytes: usize,
    pub durations: Vec<Duration>,
    pub error: Option<String>,
}

impl SweepResult {
    pub fn avg_gbps(&self) -> f64 {
        if self.durations.is_empty() {
            return 0.0;
        }
        let avg = self.durations.iter().sum::<Duration>() / self.durations.len() as u32;
        self.total_bytes as f64 / avg.as_secs_f64() / 1e9
    }

    pub fn best_gbps(&self) -> f64 {
        self.durations
            .iter()
            .min()
            .map(|d| self.total_bytes as f64 / d.as_secs_f64() / 1e9)
            .unwrap_or(0.0)
    }
}

pub async fn cmd_sweep(
    cli: &Cli,
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
    println!("Sweep: {total_configs} configurations to test");
    println!();

    let resolved = resolve_layout(cli);
    let bb = resolved.block_bytes();
    let nb = effective_num_blocks(cli);

    let mut results: Vec<SweepResult> = Vec::new();

    for (idx, point) in points.into_iter().enumerate() {
        let label = format!(
            "[{}/{}] io_api={} o_direct={} backend={} chunk_size={} conc_chunks={} agent_per_chunk={}",
            idx + 1, total_configs,
            point.io_api, point.o_direct, point.disk_backend,
            point.chunk_size, point.concurrent_chunks, point.agent_per_chunk,
        );
        eprintln!("{label}");

        unsafe {
            std::env::set_var(
                "DYN_KVBM_REMOTE_DISK_O_DIRECT",
                if point.o_direct { "true" } else { "false" },
            );
        }

        let use_gds = point.disk_backend != "posix";
        let disk_flags = match point.disk_backend.as_str() {
            "gds" => dynamo_llm::block_manager::config::DISK_FLAGS_GDS_BOTH,
            "gds-read-only" | "gds-read" => dynamo_llm::block_manager::config::DISK_FLAGS_GDS_READS_ONLY,
            _ => dynamo_llm::block_manager::config::DISK_FLAGS_POSIX_BOTH,
        };

        let per_user_per_worker = nb * bb;
        let total_bytes = per_user_per_worker * cli.tp * num_users;
        let chunk_sz = if point.chunk_size == 0 { nb } else { point.chunk_size };

        // Try to create an agent to check if the backend is available
        let agent_result = std::panic::catch_unwind(|| {
            worker::build_agent("sweep-probe", &point.io_api, use_gds)
        });

        if agent_result.is_err() {
            eprintln!("  SKIPPED (backend unavailable)");
            results.push(SweepResult {
                point,
                total_bytes,
                durations: vec![],
                error: Some("backend unavailable".into()),
            });
            continue;
        }
        drop(agent_result);

        // Build workers for this config (reused across iterations)
        let workers: Vec<Worker> = (0..cli.tp)
            .map(|wid| {
                let agent = worker::build_agent(
                    &format!("sweep-w{wid}"),
                    &point.io_api,
                    use_gds,
                );
                let mut user_layouts = Vec::with_capacity(num_users);
                let mut user_blocks = Vec::with_capacity(num_users);
                let mut user_descriptors = Vec::with_capacity(num_users);

                for uid in 0..num_users {
                    let config = worker::build_layout_config(cli, &resolved);
                    let (layout, blocks) = worker::allocate_and_register(config, &agent);
                    let descs = worker::make_descriptors(cli.bench_dir(), nb, bb, wid, cli.tp, uid);
                    user_layouts.push(layout);
                    user_blocks.push(blocks);
                    user_descriptors.push(descs);
                }

                let remote_ctx = worker::build_remote_ctx(
                    std::sync::Arc::new(Some(agent.clone())),
                    cli.bench_dir(),
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
                    let io_api = point.io_api.clone();
                    let use_gds = use_gds;
                    let disk_flags = disk_flags;
                    let wid = w.id;
                    let concurrent_chunks = point.concurrent_chunks;
                    let agent_per_chunk = point.agent_per_chunk;

                    handles.push(tokio::spawn(async move {
                        worker::run_chunked_pipeline(
                            &blocks, &descs, &ctx,
                            chunk_sz, concurrent_chunks, agent_per_chunk,
                            &io_api, use_gds, disk_flags, wid, &cancel,
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

        results.push(SweepResult {
            point,
            total_bytes,
            durations,
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
