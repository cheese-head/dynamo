// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Result;
use nixl_sys::Agent as NixlAgent;
use tokio::time::MissedTickBehavior;
use tokio_util::sync::CancellationToken;

use dynamo_llm::block_manager::{
    LayoutConfig, NixlLayout, PinnedStorage,
    block::{
        BlockData,
        transfer::{
            TransferContext,
            remote::{RemoteBlockDescriptor, RemoteTransferPipeline},
        },
    },
    config::{DiskTransferFlags, RemoteStorageConfig, RemoteTransferContext},
    layout::{BlockLayoutConfig, FullyContiguous},
    storage::{
        PinnedAllocator, Storage, StorageAllocator, SystemAllocator, SystemStorage,
        nixl::{NixlRegisterableStorage, NixlDescriptor},
    },
};

use crate::cli::DiskArgs;

/// Trait that ties a storage type to its allocator so generic bench code can
/// create the right allocator without carrying an extra type parameter.
pub trait BenchStorage: Storage + NixlRegisterableStorage + NixlDescriptor + Send + Sync + 'static {
    type Allocator: StorageAllocator<Self> + Send + Sync + 'static;
    fn create_allocator() -> Self::Allocator;
    fn name() -> &'static str;
}

impl BenchStorage for PinnedStorage {
    type Allocator = PinnedAllocator;
    fn create_allocator() -> PinnedAllocator {
        PinnedAllocator::new().expect("PinnedAllocator::new")
    }
    fn name() -> &'static str { "pinned" }
}

impl BenchStorage for SystemStorage {
    type Allocator = SystemAllocator;
    fn create_allocator() -> SystemAllocator { SystemAllocator }
    fn name() -> &'static str { "system" }
}

pub fn build_agent(name: &str, io_api: &str, use_gds: bool, gds_threads: usize) -> NixlAgent {
    let agent = NixlAgent::new(name).expect("Failed to create NIXL agent");

    if use_gds {
        match agent.get_plugin_params("GDS_MT") {
            Ok((_, default_gds_params)) => {
                let mut gds_params = default_gds_params.clone().expect("Failed to clone GDS_MT params");
                if gds_threads > 0 {
                    gds_params.set("thread_count", &gds_threads.to_string()).unwrap();
                }
                match agent.create_backend("GDS_MT", &gds_params) {
                    Ok(_) => tracing::info!("GDS_MT backend created"),
                    Err(e) => tracing::warn!(error = %e, "GDS_MT backend failed"),
                }
            },
            Err(e) => tracing::warn!(error = %e, "GDS_MT plugin not available"),
        }
    }

    let (_, default_params) = agent
        .get_plugin_params("POSIX")
        .expect("POSIX plugin not available");
    let mut params = default_params.clone().expect("Failed to clone POSIX params");

    match io_api {
        "aio" | "linux_aio" | "libaio" => {
            params.set("use_aio", "true").unwrap();
            params.set("use_uring", "false").unwrap();
            params.set("use_posix_aio", "false").unwrap();
        }
        "uring" | "io_uring" => {
            params.set("use_aio", "false").unwrap();
            params.set("use_uring", "true").unwrap();
            params.set("use_posix_aio", "false").unwrap();
        }
        _ => {}
    }

    agent
        .create_backend("POSIX", &params)
        .expect("Failed to create POSIX backend");
    agent
}

pub fn build_layout_config(cli: &DiskArgs, layout: &super::layout::ResolvedLayout) -> LayoutConfig {
    let nb = super::layout::effective_num_blocks(cli);
    let (nl, od, ps, id, dt) = layout.as_tuple();
    LayoutConfig::builder()
        .num_blocks(nb)
        .num_layers(nl)
        .outer_dim(od)
        .page_size(ps)
        .inner_dim(id)
        .dtype_width_bytes(dt)
        .build()
        .expect("Invalid layout config")
}

pub fn allocate_and_register<S: BenchStorage>(
    config: LayoutConfig,
    agent: &NixlAgent,
) -> (Arc<FullyContiguous<S>>, Vec<BlockData<S>>) {
    let allocator = S::create_allocator();
    let mut layout =
        FullyContiguous::allocate(config, &allocator).expect("FullyContiguous::allocate");
    layout.nixl_register(agent, None).expect("nixl_register failed");

    let num = layout.num_blocks();
    let layout = Arc::new(layout);
    let blocks: Vec<BlockData<S>> = (0..num)
        .map(|i| BlockData::new(layout.clone(), i, 0, 0))
        .collect();
    (layout, blocks)
}

pub fn make_descriptors(
    dir: &str,
    num_blocks: usize,
    block_bytes: usize,
    worker_id: usize,
    world_size: usize,
    user_id: usize,
) -> Vec<RemoteBlockDescriptor> {
    let base_hash = 0x1000 + (user_id as u64) * (num_blocks as u64);
    (0..num_blocks as u64)
        .map(|i| {
            RemoteBlockDescriptor::disk_from_hash(dir, base_hash + i, block_bytes, worker_id, world_size)
        })
        .collect()
}

pub fn build_remote_ctx(
    agent: Arc<Option<NixlAgent>>,
    dir: &str,
    worker_id: usize,
    world_size: usize,
    disk_flags: DiskTransferFlags,
) -> Arc<RemoteTransferContext> {
    let cuda_ctx = cudarc::driver::CudaContext::new(0).expect("CudaContext::new(0)");
    let stream = cuda_ctx.default_stream();
    let handle = tokio::runtime::Handle::current();

    let base = Arc::new(
        TransferContext::new(agent, stream, handle, None).expect("TransferContext::new"),
    );

    Arc::new(
        RemoteTransferContext::new(base, RemoteStorageConfig::disk(dir, disk_flags))
            .with_topology(worker_id as u64, world_size),
    )
}

pub struct Worker<S: Storage> {
    pub id: usize,
    pub _agent: NixlAgent,
    pub user_layouts: Vec<Arc<FullyContiguous<S>>>,
    pub user_blocks: Vec<Vec<BlockData<S>>>,
    pub user_descriptors: Vec<Vec<RemoteBlockDescriptor>>,
    pub remote_ctx: Arc<RemoteTransferContext>,
}

impl<S: BenchStorage> Worker<S> {
    pub fn new(cli: &DiskArgs, resolved: &super::layout::ResolvedLayout, worker_id: usize, num_users: usize) -> Self {
        let agent = build_agent(&format!("bench-worker-{worker_id}"), &cli.io_api, cli.use_gds(), cli.gds_threads);
        let bb = resolved.block_bytes();
        let nb = super::layout::effective_num_blocks(cli);

        let mut user_layouts = Vec::with_capacity(num_users);
        let mut user_blocks = Vec::with_capacity(num_users);
        let mut user_descriptors = Vec::with_capacity(num_users);

        for uid in 0..num_users {
            let config = build_layout_config(cli, resolved);
            let (layout, blocks) = allocate_and_register::<S>(config, &agent);
            let descs = make_descriptors(cli.bench_dir(), nb, bb, worker_id, cli.tp, uid);
            user_layouts.push(layout);
            user_blocks.push(blocks);
            user_descriptors.push(descs);
        }

        let remote_ctx = build_remote_ctx(
            Arc::new(Some(agent.clone())),
            cli.bench_dir(),
            worker_id,
            cli.tp,
            cli.disk_transfer_flags(),
        );

        Self {
            id: worker_id,
            _agent: agent,
            user_layouts,
            user_blocks,
            user_descriptors,
            remote_ctx,
        }
    }
}

pub async fn run_chunked_pipeline<S: BenchStorage>(
    blocks: &[BlockData<S>],
    descriptors: &[RemoteBlockDescriptor],
    remote_ctx: &Arc<RemoteTransferContext>,
    chunk_size: usize,
    concurrent_chunks: usize,
    agent_per_chunk: bool,
    agent_pool_size: usize,
    io_api: &str,
    use_gds: bool,
    gds_threads: usize,
    disk_flags: DiskTransferFlags,
    worker_id: usize,
    cancel: &CancellationToken,
) -> Result<()> {
    let num_blocks = descriptors.len();
    let chunk_sz = if chunk_size == 0 { num_blocks } else { chunk_size };
    let num_chunks = (num_blocks + chunk_sz - 1) / chunk_sz;

    let pool: Option<Vec<Arc<RemoteTransferContext>>> = if agent_pool_size > 0 {
        let pool_ctxs: Vec<Arc<RemoteTransferContext>> = (0..agent_pool_size)
            .map(|i| {
                let a = build_agent(
                    &format!("bench-w{worker_id}-pool{i}"),
                    io_api,
                    use_gds,
                    gds_threads,
                );
                build_remote_ctx(
                    Arc::new(Some(a)),
                    remote_ctx.base_path().unwrap_or("/tmp"),
                    remote_ctx.worker_id() as usize,
                    remote_ctx.world_size(),
                    disk_flags,
                )
            })
            .collect();
        Some(pool_ctxs)
    } else {
        None
    };

    if concurrent_chunks == 0 {
        for chunk_idx in 0..num_chunks {
            let s = chunk_idx * chunk_sz;
            let e = (s + chunk_sz).min(num_blocks);
            let ctx = match &pool {
                Some(p) => p[chunk_idx % p.len()].clone(),
                None => remote_ctx.clone(),
            };
            let sub = RemoteTransferPipeline::onboard_direct(descriptors[s..e].to_vec());
            sub.execute(&blocks[s..e], ctx.as_ref(), cancel)
                .await
                .map_err(|e| anyhow::anyhow!("worker {worker_id} chunk {chunk_idx}: {e}"))?;
        }
    } else {
        let sem = Arc::new(tokio::sync::Semaphore::new(concurrent_chunks));
        let (done_tx, mut done_rx) =
            tokio::sync::mpsc::unbounded_channel::<Result<(), String>>();

        for chunk_idx in 0..num_chunks {
            let s = chunk_idx * chunk_sz;
            let e = (s + chunk_sz).min(num_blocks);

            let chunk_descs = descriptors[s..e].to_vec();
            let chunk_blocks = blocks[s..e].to_vec();

            let ctx = if let Some(ref p) = pool {
                p[chunk_idx % p.len()].clone()
            } else if agent_per_chunk {
                let a = build_agent(
                    &format!("bench-w{worker_id}-c{chunk_idx}"),
                    io_api,
                    use_gds,
                    gds_threads,
                );
                build_remote_ctx(
                    Arc::new(Some(a)),
                    remote_ctx.base_path().unwrap_or("/tmp"),
                    remote_ctx.worker_id() as usize,
                    remote_ctx.world_size(),
                    disk_flags,
                )
            } else {
                remote_ctx.clone()
            };

            let cancel = cancel.clone();
            let sem = sem.clone();
            let done_tx = done_tx.clone();

            tokio::spawn(async move {
                let _permit = sem.acquire().await.unwrap();
                let sub = RemoteTransferPipeline::onboard_direct(chunk_descs);
                let result = sub
                    .execute(&chunk_blocks, ctx.as_ref(), &cancel)
                    .await
                    .map_err(|e| format!("worker {worker_id} chunk {chunk_idx}: {e}"));
                let _ = done_tx.send(result);
            });
        }
        drop(done_tx);

        let mut first_err: Option<String> = None;
        while let Some(result) = done_rx.recv().await {
            if let Err(e) = result {
                if first_err.is_none() {
                    first_err = Some(e);
                }
            }
        }
        if let Some(e) = first_err {
            anyhow::bail!("{e}");
        }
    }

    Ok(())
}

pub fn spawn_progress_heartbeat(interval_sec: u64, label: String, start: Instant, cancel: &CancellationToken) {
    if interval_sec == 0 {
        return;
    }
    let c = cancel.clone();
    let dur = Duration::from_secs(interval_sec);
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(dur);
        interval.set_missed_tick_behavior(MissedTickBehavior::Skip);
        loop {
            tokio::select! {
                _ = c.cancelled() => break,
                _ = interval.tick() => {
                    eprintln!("  [{}] {:.1}s elapsed …", label, start.elapsed().as_secs_f64());
                }
            }
        }
    });
}
