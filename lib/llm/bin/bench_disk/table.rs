// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::io::Write;
use std::time::Duration;

use comfy_table::{Cell, Color, ContentArrangement, Table, presets::UTF8_FULL_CONDENSED};

use crate::sweep::SweepResult;

pub fn print_sweep_table(results: &[SweepResult]) {
    let mut table = Table::new();
    table.load_preset(UTF8_FULL_CONDENSED);
    table.set_content_arrangement(ContentArrangement::Dynamic);

    table.set_header(vec![
        "#", "io_api", "O_DIRECT", "backend", "chunk_sz",
        "conc_chunks", "agent_per", "avg GB/s", "best GB/s", "avg elapsed", "status",
    ]);

    for (i, r) in results.iter().enumerate() {
        let status = if let Some(e) = &r.error {
            format!("ERR: {}", &e[..e.len().min(30)])
        } else {
            "OK".into()
        };

        let avg_gbps = if r.durations.is_empty() {
            "-".into()
        } else {
            format!("{:.2}", r.avg_gbps())
        };

        let best_gbps = if r.durations.is_empty() {
            "-".into()
        } else {
            format!("{:.2}", r.best_gbps())
        };

        let avg_elapsed = if r.durations.is_empty() {
            "-".into()
        } else {
            let avg = r.durations.iter().sum::<Duration>() / r.durations.len() as u32;
            format!("{:.3}s", avg.as_secs_f64())
        };

        let row = vec![
            Cell::new(i + 1),
            Cell::new(&r.point.io_api),
            Cell::new(r.point.o_direct),
            Cell::new(&r.point.disk_backend),
            Cell::new(r.point.chunk_size),
            Cell::new(if r.point.concurrent_chunks == 0 { "serial".into() } else { r.point.concurrent_chunks.to_string() }),
            Cell::new(if r.point.agent_per_chunk { "yes" } else { "no" }),
            Cell::new(&avg_gbps).fg(if r.avg_gbps() > 1.0 { Color::Green } else { Color::Reset }),
            Cell::new(&best_gbps).fg(if r.best_gbps() > 1.0 { Color::Green } else { Color::Reset }),
            Cell::new(&avg_elapsed),
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
    writeln!(f, "io_api,o_direct,backend,chunk_size,concurrent_chunks,agent_per_chunk,avg_gbps,best_gbps,avg_elapsed_s,status")?;

    for r in results {
        let status = if let Some(e) = &r.error { e.as_str() } else { "ok" };
        let avg = if r.durations.is_empty() {
            0.0
        } else {
            let d = r.durations.iter().sum::<Duration>() / r.durations.len() as u32;
            d.as_secs_f64()
        };
        writeln!(
            f,
            "{},{},{},{},{},{},{:.4},{:.4},{:.4},{}",
            r.point.io_api, r.point.o_direct, r.point.disk_backend,
            r.point.chunk_size, r.point.concurrent_chunks, r.point.agent_per_chunk,
            r.avg_gbps(), r.best_gbps(), avg, status,
        )?;
    }
    Ok(())
}
