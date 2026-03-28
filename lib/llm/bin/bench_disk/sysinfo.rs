// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::fs;
use std::path::Path;

pub fn print_system_info(bench_dir: &str) {
    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║  System Information                                                  ║");
    println!("╠══════════════════════════════════════════════════════════════════════╣");

    print_host_info();
    print_gpu_info();
    print_network_info();
    print_pcie_info();
    print_storage_info(bench_dir);
    print_nixl_info();

    println!("╚══════════════════════════════════════════════════════════════════════╝");
    println!();
}

fn print_host_info() {
    let hostname = fs::read_to_string("/etc/hostname")
        .unwrap_or_else(|_| "unknown".into())
        .trim()
        .to_string();
    println!("║  Hostname:     {:<55}║", hostname);

    let os = read_os_release();
    let kernel = read_file_trimmed("/proc/sys/kernel/osrelease");
    println!("║  OS:           {:<55}║", format!("{os} (kernel {kernel})"));

    let (cpu_model, cpu_count) = read_cpu_info();
    println!("║  CPU:          {:<55}║", format!("{cpu_model} ({cpu_count} threads)"));

    let mem_gb = read_mem_total_gb();
    println!("║  Memory:       {:<55}║", format!("{mem_gb:.1} GB"));
}

fn print_gpu_info() {
    println!("╠──────────────────────────────────────────────────────────────────────╣");
    println!("║  GPU                                                                 ║");

    let gpu_count = cudarc::driver::CudaContext::device_count().unwrap_or(0);
    println!("║  Count:        {:<55}║", gpu_count);

    if let Ok(output) = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"])
        .output()
    {
        if let Ok(stdout) = String::from_utf8(output.stdout) {
            for (i, line) in stdout.lines().enumerate() {
                let parts: Vec<&str> = line.split(", ").collect();
                if parts.len() >= 3 {
                    if i == 0 {
                        println!("║  GPU 0:        {:<55}║", format!("{} ({}MB)", parts[0].trim(), parts[1].trim()));
                        println!("║  Driver:       {:<55}║", parts[2].trim());
                    }
                }
            }
        }
    }

    let cuda_ver = read_file_trimmed("/usr/local/cuda/version.json")
        .chars()
        .take(0)
        .collect::<String>();
    if let Ok(output) = std::process::Command::new("nvcc").args(["--version"]).output() {
        if let Ok(stdout) = String::from_utf8(output.stdout) {
            for line in stdout.lines() {
                if line.contains("release") {
                    if let Some(ver) = line.split("release ").nth(1) {
                        let ver = ver.split(',').next().unwrap_or(ver);
                        println!("║  CUDA:         {:<55}║", ver.trim());
                    }
                }
            }
        }
    } else if !cuda_ver.is_empty() {
        println!("║  CUDA:         {:<55}║", cuda_ver);
    }
}

fn print_network_info() {
    println!("╠──────────────────────────────────────────────────────────────────────╣");
    println!("║  Network                                                             ║");

    let ib_path = Path::new("/sys/class/infiniband");
    if ib_path.exists() {
        if let Ok(entries) = fs::read_dir(ib_path) {
            for entry in entries.flatten() {
                let name = entry.file_name().to_string_lossy().to_string();
                let board_id = read_file_trimmed(&format!("/sys/class/infiniband/{name}/board_id"));
                let rate = read_file_trimmed(&format!("/sys/class/infiniband/{name}/ports/1/rate"));
                let pcie = read_pcie_bdf_for_ib(&name);
                println!("║  {:<12}   {:<55}║",
                    format!("{name}:"),
                    format!("{board_id} {rate} {pcie}"),
                );
            }
        }
    }

    let net_path = Path::new("/sys/class/net");
    if let Ok(entries) = fs::read_dir(net_path) {
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            if name == "lo" {
                continue;
            }
            let speed = read_file_trimmed(&format!("/sys/class/net/{name}/speed"));
            let mtu = read_file_trimmed(&format!("/sys/class/net/{name}/mtu"));
            let operstate = read_file_trimmed(&format!("/sys/class/net/{name}/operstate"));
            if operstate == "up" || operstate == "unknown" {
                println!("║  {:<12}   {:<55}║",
                    format!("{name}:"),
                    format!("{speed}Mb/s MTU {mtu} ({operstate})"),
                );
            }
        }
    }
}

fn print_pcie_info() {
    println!("╠──────────────────────────────────────────────────────────────────────╣");
    println!("║  PCIe Topology                                                       ║");

    if let Ok(output) = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=pci.bus_id,pcie.link.gen.current,pcie.link.width.current", "--format=csv,noheader"])
        .output()
    {
        if let Ok(stdout) = String::from_utf8(output.stdout) {
            for (i, line) in stdout.lines().enumerate() {
                let parts: Vec<&str> = line.split(", ").collect();
                if parts.len() >= 3 {
                    let bdf = parts[0].trim();
                    let numa = read_file_trimmed(&format!(
                        "/sys/bus/pci/devices/{}/numa_node",
                        bdf.to_lowercase()
                    ));
                    println!("║  GPU {:<9} {:<55}║",
                        format!("{i}:"),
                        format!("{bdf} Gen{} x{} NUMA {numa}", parts[1].trim(), parts[2].trim()),
                    );
                }
            }
        }
    }

    if let Ok(entries) = fs::read_dir("/sys/class/nvme") {
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            let model = read_file_trimmed(&format!("/sys/class/nvme/{name}/model"));
            let addr = read_file_trimmed(&format!("/sys/class/nvme/{name}/address"));
            if !model.is_empty() {
                println!("║  {:<12}   {:<55}║", format!("{name}:"), format!("{model} @ {addr}"));
            }
        }
    }
}

fn print_storage_info(bench_dir: &str) {
    println!("╠──────────────────────────────────────────────────────────────────────╣");
    println!("║  Storage                                                             ║");
    println!("║  Path:         {:<55}║", bench_dir);

    if let Ok(mounts) = fs::read_to_string("/proc/mounts") {
        let mut best_match: Option<(&str, &str, &str, &str)> = None;
        let mut best_len = 0;

        for line in mounts.lines() {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 4 {
                let mount_point = parts[1];
                if bench_dir.starts_with(mount_point) && mount_point.len() > best_len {
                    best_len = mount_point.len();
                    best_match = Some((parts[0], parts[1], parts[2], parts[3]));
                }
            }
        }

        if let Some((dev, _mount, fstype, opts)) = best_match {
            println!("║  Filesystem:   {:<55}║", format!("{dev} ({fstype})"));
            let short_opts: String = opts.chars().take(50).collect();
            println!("║  Mount opts:   {:<55}║", short_opts);
        }
    }

    if let Ok(stat) = nix::sys::statvfs::statvfs(bench_dir) {
        let total = stat.blocks() as f64 * stat.fragment_size() as f64 / 1e12;
        let avail = stat.blocks_available() as f64 * stat.fragment_size() as f64 / 1e12;
        println!("║  Total/Avail:  {:<55}║", format!("{total:.1}T / {avail:.1}T"));
    }
}

fn print_nixl_info() {
    println!("╠──────────────────────────────────────────────────────────────────────╣");
    println!("║  NIXL / UCX                                                          ║");

    let ucx_tls = std::env::var("UCX_TLS").unwrap_or_else(|_| "(system default)".into());
    println!("║  UCX TLS:      {:<55}║", ucx_tls);
}

fn read_file_trimmed(path: &str) -> String {
    fs::read_to_string(path)
        .unwrap_or_default()
        .trim()
        .to_string()
}

fn read_os_release() -> String {
    if let Ok(contents) = fs::read_to_string("/etc/os-release") {
        for line in contents.lines() {
            if let Some(val) = line.strip_prefix("PRETTY_NAME=") {
                return val.trim_matches('"').to_string();
            }
        }
    }
    "Linux".into()
}

fn read_cpu_info() -> (String, usize) {
    let mut model = String::from("unknown");
    let mut count = 0usize;
    if let Ok(contents) = fs::read_to_string("/proc/cpuinfo") {
        for line in contents.lines() {
            if line.starts_with("model name") {
                if let Some(val) = line.split(':').nth(1) {
                    model = val.trim().to_string();
                }
            }
            if line.starts_with("processor") {
                count += 1;
            }
        }
    }
    (model, count)
}

fn read_mem_total_gb() -> f64 {
    if let Ok(contents) = fs::read_to_string("/proc/meminfo") {
        for line in contents.lines() {
            if line.starts_with("MemTotal:") {
                if let Some(kb_str) = line.split_whitespace().nth(1) {
                    if let Ok(kb) = kb_str.parse::<f64>() {
                        return kb / 1_048_576.0;
                    }
                }
            }
        }
    }
    0.0
}

fn read_pcie_bdf_for_ib(ib_name: &str) -> String {
    let device_link = format!("/sys/class/infiniband/{ib_name}/device");
    if let Ok(target) = fs::read_link(&device_link) {
        if let Some(bdf) = target.file_name() {
            return bdf.to_string_lossy().to_string();
        }
    }
    String::new()
}
