// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::io::Write;
use std::time::Duration;

use comfy_table::{Cell, Color, ContentArrangement, Table, presets::UTF8_FULL_CONDENSED};

use crate::sweep::SweepResult;

fn fmt_gbps(total_bytes: usize, d: Duration) -> String {
    format!("{:.2}", total_bytes as f64 / d.as_secs_f64() / 1e9)
}

pub fn print_sweep_table(results: &[SweepResult]) {
    let mut table = Table::new();
    table.load_preset(UTF8_FULL_CONDENSED);
    table.set_content_arrangement(ContentArrangement::Dynamic);

    table.set_header(vec![
        "#", "io_api", "backend", "o_direct",
        "chunk", "conc", "users", "poll_us", "apc", "pool", "rt_thr",
        "wr GB/s", "rd GB/s",
        "p50", "p95", "p99",
        "min", "max",
        "fd p99",
        "status",
    ]);

    for (i, r) in results.iter().enumerate() {
        let status = if let Some(e) = &r.error {
            format!("ERR: {}", &e[..e.len().min(30)])
        } else {
            "OK".into()
        };

        let rt_threads = if r.point.runtime_threads == 0 { "def".into() } else { r.point.runtime_threads.to_string() };
        let conc = if r.point.concurrent_chunks == 0 { "seq".into() } else { r.point.concurrent_chunks.to_string() };
        let apc = if r.point.agent_per_chunk { "Y" } else { "-" };
        let pool = if r.point.agent_pool_size == 0 { "-".into() } else { r.point.agent_pool_size.to_string() };

        let write_gbps = match r.write_duration {
            Some(d) if d.as_nanos() > 0 => fmt_gbps(r.total_bytes, d),
            _ => "-".into(),
        };

        let fd_open_p95 = match r.fd_open_p99 {
            Some(d) => format!("{:.1}ms", d.as_secs_f64() * 1000.0),
            None => "-".into(),
        };

        let stats = r.read_stats();
        let read_avg = stats.as_ref().map(|s| fmt_gbps(r.total_bytes, s.avg)).unwrap_or("-".into());
        let read_p50 = stats.as_ref().map(|s| fmt_gbps(r.total_bytes, s.p50)).unwrap_or("-".into());
        let read_p95 = stats.as_ref().map(|s| fmt_gbps(r.total_bytes, s.p95)).unwrap_or("-".into());
        let read_p99 = stats.as_ref().map(|s| fmt_gbps(r.total_bytes, s.p99)).unwrap_or("-".into());
        let read_min = stats.as_ref().map(|s| fmt_gbps(r.total_bytes, s.max)).unwrap_or("-".into());
        let read_max = stats.as_ref().map(|s| fmt_gbps(r.total_bytes, s.min)).unwrap_or("-".into());

        let users = if r.point.users == 0 { "cli".into() } else { r.point.users.to_string() };
        let poll_us = if r.point.nixl_poll_interval_us == 0 {
            "-".into()
        } else {
            r.point.nixl_poll_interval_us.to_string()
        };

        let color = if r.avg_gbps() > 1.0 { Color::Green } else { Color::Reset };
        let row = vec![
            Cell::new(i + 1),
            Cell::new(&r.point.nixl_posix_api),
            Cell::new(&r.point.remote_disk_backend),
            Cell::new(r.point.remote_disk_o_direct),
            Cell::new(r.point.chunk_size),
            Cell::new(&conc),
            Cell::new(&users),
            Cell::new(&poll_us),
            Cell::new(apc),
            Cell::new(&pool),
            Cell::new(&rt_threads),
            Cell::new(&write_gbps),
            Cell::new(&read_avg).fg(color),
            Cell::new(&read_p50).fg(color),
            Cell::new(&read_p95).fg(color),
            Cell::new(&read_p99).fg(color),
            Cell::new(&read_min),
            Cell::new(&read_max),
            Cell::new(&fd_open_p95),
            Cell::new(&status),
        ];
        table.add_row(row);
    }

    println!("{table}");
}

pub fn print_results(label: &str, total_bytes: usize, durations: &[Duration]) {
    if durations.is_empty() {
        return;
    }
    let avg = durations.iter().sum::<Duration>() / durations.len() as u32;
    let tp = total_bytes as f64 / avg.as_secs_f64() / 1e9;
    let min = durations.iter().min().unwrap();
    let max = durations.iter().max().unwrap();
    let min_tp = total_bytes as f64 / max.as_secs_f64() / 1e9;
    let max_tp = total_bytes as f64 / min.as_secs_f64() / 1e9;

    println!();
    println!("=== {label} ===");
    println!("  Total data:     {:.2} GB", total_bytes as f64 / 1e9);
    println!("  Iterations:     {}", durations.len());
    println!("  Avg elapsed:    {:.3} s", avg.as_secs_f64());
    println!("  Min elapsed:    {:.3} s", min.as_secs_f64());
    println!("  Max elapsed:    {:.3} s", max.as_secs_f64());
    println!("  Avg throughput: {tp:.2} GB/s");
    println!("  Min throughput: {min_tp:.2} GB/s");
    println!("  Max throughput: {max_tp:.2} GB/s");
}

pub fn write_csv(results: &[SweepResult], path: &str) -> anyhow::Result<()> {
    let mut f = std::fs::File::create(path)?;
    writeln!(f, "nixl_posix_api,remote_disk_o_direct,remote_disk_backend,runtime_threads,\
        chunk_size,concurrent_chunks,agent_per_chunk,agent_pool_size,users,nixl_poll_interval_us,\
        write_gbps,write_s,\
        read_avg_gbps,read_p50_gbps,read_p95_gbps,read_p99_gbps,read_min_gbps,read_max_gbps,\
        read_avg_s,read_p50_s,read_p95_s,read_p99_s,read_min_s,read_max_s,\
        fd_open_p99_ms,iterations,status")?;

    for r in results {
        let status = if let Some(e) = &r.error { e.as_str() } else { "ok" };
        let tb = r.total_bytes as f64;

        let (w_gbps, w_s) = match r.write_duration {
            Some(d) if d.as_nanos() > 0 => (tb / d.as_secs_f64() / 1e9, d.as_secs_f64()),
            _ => (0.0, 0.0),
        };
        let fd_p99_ms = r.fd_open_p99.map(|d| d.as_secs_f64() * 1000.0).unwrap_or(0.0);

        let stats = r.read_stats();
        let (avg_gbps, p50_gbps, p95_gbps, p99_gbps, min_gbps, max_gbps) = match &stats {
            Some(s) => (
                s.gbps(r.total_bytes), tb / s.p50.as_secs_f64() / 1e9,
                tb / s.p95.as_secs_f64() / 1e9, tb / s.p99.as_secs_f64() / 1e9,
                tb / s.max.as_secs_f64() / 1e9, tb / s.min.as_secs_f64() / 1e9,
            ),
            None => (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        };
        let (avg_s, p50_s, p95_s, p99_s, min_s, max_s) = match &stats {
            Some(s) => (
                s.avg.as_secs_f64(), s.p50.as_secs_f64(),
                s.p95.as_secs_f64(), s.p99.as_secs_f64(),
                s.min.as_secs_f64(), s.max.as_secs_f64(),
            ),
            None => (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        };

        writeln!(
            f,
            "{},{},{},{},{},{},{},{},{},{},\
            {:.4},{:.4},\
            {:.4},{:.4},{:.4},{:.4},{:.4},{:.4},\
            {:.4},{:.4},{:.4},{:.4},{:.4},{:.4},\
            {:.2},{},{}",
            r.point.nixl_posix_api, r.point.remote_disk_o_direct, r.point.remote_disk_backend,
            r.point.runtime_threads,
            r.point.chunk_size, r.point.concurrent_chunks, r.point.agent_per_chunk,
            r.point.agent_pool_size, r.point.users, r.point.nixl_poll_interval_us,
            w_gbps, w_s,
            avg_gbps, p50_gbps, p95_gbps, p99_gbps, min_gbps, max_gbps,
            avg_s, p50_s, p95_s, p99_s, min_s, max_s,
            fd_p99_ms, r.read_durations.len(), status,
        )?;
    }
    Ok(())
}
