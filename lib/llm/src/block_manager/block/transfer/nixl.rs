// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

use super::remote::{RemoteBlockDescriptor, RemoteKey, RemoteStorageKind, RemoteTransferDirection};
use crate::block_manager::config::RemoteTransferContext;
use crate::block_manager::storage::nixl::NixlRegisterableStorage;
use crate::block_manager::storage::{ObjectStorage, RemoteDiskStorage};
use anyhow::Result;
use nixl_sys::{
    Agent as NixlAgent, MemType, MemoryRegion, NixlDescriptor, OptArgs, XferDescList, XferOp,
    XferRequest, XferStatus,
};
use once_cell::sync::Lazy;
use parking_lot::Mutex as SyncMutex;
use std::collections::HashMap;
use std::future::Future;
use std::time::Duration;
use tokio::sync::Mutex as AsyncMutex;
use tokio_util::sync::CancellationToken;

const DEFAULT_REMOTE_DISK_FD_CACHE_MAX_ENTRIES: usize = 131_072;
const REMOTE_DISK_O_DIRECT_KEY: &str = "DYN_KVBM_REMOTE_DISK_O_DIRECT";
const REMOTE_DISK_ALIGNMENT_VALIDATE_KEY: &str = "DYN_KVBM_REMOTE_DISK_VALIDATE_ALIGNMENT";
const REMOTE_DISK_ALIGNMENT_OVERRIDE_KEY: &str = "DYN_KVBM_REMOTE_DISK_ALIGNMENT_BYTES";
const DEFAULT_O_DIRECT_ALIGNMENT_FALLBACK: usize = 4096;

static REMOTE_DISK_FD_CACHE_MAX_ENTRIES: Lazy<usize> = Lazy::new(|| {
    std::env::var("DYN_KVBM_REMOTE_DISK_FD_CACHE_MAX_ENTRIES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_REMOTE_DISK_FD_CACHE_MAX_ENTRIES)
});

static REMOTE_DISK_FD_CACHE: Lazy<AsyncMutex<RemoteDiskFdCache>> =
    Lazy::new(|| AsyncMutex::new(RemoteDiskFdCache::new(*REMOTE_DISK_FD_CACHE_MAX_ENTRIES)));

#[derive(Debug, Clone, Copy)]
struct RemoteDiskAlignmentConfig {
    quantum: usize,
    validate: bool,
}

static REMOTE_DISK_ALIGNMENT_CONFIG: Lazy<RemoteDiskAlignmentConfig> = Lazy::new(|| {
    let page_size = nix::unistd::sysconf(nix::unistd::SysconfVar::PAGE_SIZE)
        .ok()
        .flatten()
        .and_then(|v| usize::try_from(v).ok())
        .filter(|v| *v > 0)
        .unwrap_or(DEFAULT_O_DIRECT_ALIGNMENT_FALLBACK);

    let alignment_override = std::env::var(REMOTE_DISK_ALIGNMENT_OVERRIDE_KEY)
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|v| *v > 0);

    let quantum = alignment_override.unwrap_or(page_size);
    let validate = std::env::var(REMOTE_DISK_ALIGNMENT_VALIDATE_KEY)
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);

    tracing::info!(
        target: "kvbm-g4",
        page_size,
        alignment_quantum = quantum,
        validate_alignment = validate,
        "remote disk alignment config initialized"
    );

    RemoteDiskAlignmentConfig { quantum, validate }
});

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct RemoteDiskFdCacheKey {
    path: String,
    use_odirect: bool,
}

#[derive(Debug)]
struct RemoteDiskFdCacheEntry {
    storage: Arc<SyncMutex<RemoteDiskStorage>>,
    last_access_tick: u64,
}

#[derive(Debug)]
struct RemoteDiskFdCache {
    entries: HashMap<RemoteDiskFdCacheKey, RemoteDiskFdCacheEntry>,
    access_tick: u64,
    max_entries: usize,
}

impl RemoteDiskFdCache {
    fn new(max_entries: usize) -> Self {
        Self {
            entries: HashMap::new(),
            access_tick: 0,
            max_entries,
        }
    }

    fn next_tick(&mut self) -> u64 {
        self.access_tick = self.access_tick.wrapping_add(1);
        self.access_tick
    }

    fn get(&mut self, key: &RemoteDiskFdCacheKey) -> Option<Arc<SyncMutex<RemoteDiskStorage>>> {
        let tick = self.next_tick();
        let entry = self.entries.get_mut(key)?;
        entry.last_access_tick = tick;
        Some(entry.storage.clone())
    }

    fn insert(
        &mut self,
        key: RemoteDiskFdCacheKey,
        storage: Arc<SyncMutex<RemoteDiskStorage>>,
    ) -> Arc<SyncMutex<RemoteDiskStorage>> {
        if self.max_entries > 0 && self.entries.len() >= self.max_entries {
            self.evict_lru();
        }

        let tick = self.next_tick();
        self.entries.insert(
            key,
            RemoteDiskFdCacheEntry {
                storage: storage.clone(),
                last_access_tick: tick,
            },
        );
        storage
    }

    fn evict_lru(&mut self) {
        let Some(lru_key) = self
            .entries
            .iter()
            .min_by_key(|(_, entry)| entry.last_access_tick)
            .map(|(k, _)| k.clone())
        else {
            return;
        };

        self.entries.remove(&lru_key);
    }

    fn clear(&mut self) {
        self.entries.clear();
        self.access_tick = 0;
    }
}

/// Drop all cached remote-disk file descriptors and their NIXL registrations.
///
/// Call this between benchmark sweep points or whenever the NIXL agent/backend
/// changes so that subsequent transfers re-open and re-register files against
/// the current agent.
pub async fn clear_remote_disk_fd_cache() {
    REMOTE_DISK_FD_CACHE.lock().await.clear();
}

static FD_OPEN_DURATIONS: Lazy<parking_lot::Mutex<Vec<Duration>>> =
    Lazy::new(|| parking_lot::Mutex::new(Vec::new()));

/// Drain and return all recorded per-file open+register durations.
pub fn take_fd_open_durations() -> Vec<Duration> {
    std::mem::take(&mut *FD_OPEN_DURATIONS.lock())
}

async fn get_or_open_remote_disk_storage(
    agent: &NixlAgent,
    path: &str,
    block_size: usize,
    create: bool,
    use_odirect: bool,
    preallocate: bool,
) -> Result<Arc<SyncMutex<RemoteDiskStorage>>, TransferError> {
    let key = RemoteDiskFdCacheKey {
        path: path.to_string(),
        use_odirect,
    };

    {
        let mut cache = REMOTE_DISK_FD_CACHE.lock().await;
        if let Some(storage) = cache.get(&key) {
            return Ok(storage);
        }
    }

    let fd_start = std::time::Instant::now();
    let mut storage =
        RemoteDiskStorage::open(path, block_size, create, use_odirect, preallocate).map_err(|e| {
            TransferError::ExecutionError(format!(
                "Failed to {} RemoteDiskStorage at {}: {:?}",
                if create { "create" } else { "open" },
                path,
                e
            ))
        })?;
    storage.nixl_register(agent, None).map_err(|e| {
        TransferError::ExecutionError(format!("Failed to register disk storage {}: {:?}", path, e))
    })?;
    FD_OPEN_DURATIONS.lock().push(fd_start.elapsed());

    let storage = Arc::new(SyncMutex::new(storage));

    let mut cache = REMOTE_DISK_FD_CACHE.lock().await;
    if let Some(existing) = cache.get(&key) {
        return Ok(existing);
    }

    Ok(cache.insert(key, storage))
}

/// Poll transfer status inline with cancellation support.
///
/// This is the fallback path when async notification registration fails.
/// Polls the agent for transfer status with a 1ms interval.
async fn poll_transfer_completion_inline(
    agent: &NixlAgent,
    xfer_req: &XferRequest,
    cancel_token: &CancellationToken,
) -> Result<(), TransferError> {
    loop {
        tokio::select! {
            _ = cancel_token.cancelled() => {
                return Err(TransferError::Cancelled);
            }
            _ = tokio::time::sleep(Duration::from_millis(1)) => {
                match agent.get_xfer_status(xfer_req) {
                    Ok(XferStatus::Success) => return Ok(()),
                    Ok(XferStatus::InProgress) => continue,
                    Err(e) => {
                        return Err(TransferError::ExecutionError(
                            format!("Transfer status check failed: {}", e)
                        ));
                    }
                }
            }
        }
    }
}

fn append_xfer_request<Source, Destination>(
    src: &Source,
    dst: &mut Destination,
    src_dl: &mut XferDescList,
    dst_dl: &mut XferDescList,
) -> Result<()>
where
    Source: BlockDataProvider,
    Source::StorageType: NixlDescriptor,
    Destination: BlockDataProviderMut,
    Destination::StorageType: NixlDescriptor,
{
    let src_data = src.block_data();
    let dst_data = dst.block_data_mut();

    if src_data.is_fully_contiguous() && dst_data.is_fully_contiguous() {
        let src_desc = src_data.block_view()?.as_nixl_descriptor();
        let dst_desc = dst_data.block_view_mut()?.as_nixl_descriptor_mut();

        unsafe {
            src_dl.add_desc(
                src_desc.as_ptr() as usize,
                src_desc.size(),
                src_desc.device_id(),
            );

            dst_dl.add_desc(
                dst_desc.as_ptr() as usize,
                dst_desc.size(),
                dst_desc.device_id(),
            );
        }

        Ok(())
    } else {
        assert_eq!(src_data.num_layers(), dst_data.num_layers());
        for layer_idx in 0..src_data.num_layers() {
            for outer_idx in 0..src_data.num_outer_dims() {
                let src_view = src_data.layer_view(layer_idx, outer_idx)?;
                let mut dst_view = dst_data.layer_view_mut(layer_idx, outer_idx)?;

                debug_assert_eq!(src_view.size(), dst_view.size());

                let src_desc = src_view.as_nixl_descriptor();
                let dst_desc = dst_view.as_nixl_descriptor_mut();

                unsafe {
                    src_dl.add_desc(
                        src_desc.as_ptr() as usize,
                        src_desc.size(),
                        src_desc.device_id(),
                    );

                    dst_dl.add_desc(
                        dst_desc.as_ptr() as usize,
                        dst_desc.size(),
                        dst_desc.device_id(),
                    );
                }
            }
        }
        Ok(())
    }
}

/// Copy a block from a source to a destination using CUDA memcpy
pub fn write_blocks_to<Source, Destination>(
    src: &[Source],
    dst: &mut [Destination],
    ctx: &Arc<TransferContext>,
    transfer_type: NixlTransfer,
) -> Result<Box<dyn Future<Output = ()> + Send + Sync + Unpin>>
where
    Source: BlockDataProvider,
    Source::StorageType: NixlDescriptor,
    Destination: BlockDataProviderMut,
    Destination::StorageType: NixlDescriptor,
{
    if src.is_empty() || dst.is_empty() {
        return Ok(Box::new(std::future::ready(())));
    }
    assert_eq!(src.len(), dst.len());

    let nixl_agent_arc = ctx.as_ref().nixl_agent();
    let nixl_agent = nixl_agent_arc
        .as_ref()
        .as_ref()
        .expect("NIXL agent not found");

    let src_mem_type = src
        .first()
        .unwrap()
        .block_data()
        .storage_type()
        .nixl_mem_type();
    let dst_mem_type = dst
        .first()
        .unwrap()
        .block_data()
        .storage_type()
        .nixl_mem_type();

    let mut src_dl = XferDescList::new(src_mem_type)?;
    let mut dst_dl = XferDescList::new(dst_mem_type)?;

    for (src, dst) in src.iter().zip(dst.iter_mut()) {
        append_xfer_request(src, dst, &mut src_dl, &mut dst_dl)?;
    }

    let xfer_req = nixl_agent.create_xfer_req(
        transfer_type.as_xfer_op(),
        &src_dl,
        &dst_dl,
        &nixl_agent.name(),
        None,
    )?;

    let still_pending = nixl_agent.post_xfer_req(&xfer_req, None)?;

    if still_pending {
        Ok(Box::new(Box::pin(async move {
            let nixl_agent = nixl_agent_arc
                .as_ref()
                .as_ref()
                .expect("NIXL agent not found");

            loop {
                match nixl_agent.get_xfer_status(&xfer_req) {
                    Ok(XferStatus::Success) => break, // Transfer is complete.
                    Ok(XferStatus::InProgress) => {
                        tokio::time::sleep(std::time::Duration::from_millis(5)).await
                    } // Transfer is still in progress.
                    Err(e) => {
                        tracing::error!("Error getting transfer status: {}", e);
                        break;
                    }
                }
            }
        })))
    } else {
        Ok(Box::new(std::future::ready(())))
    }
}

/// Execute a remote storage transfer (object storage or disk).
///
/// This function handles the NIXL-level execution of remote transfers.
/// It supports both object storage and remote disk.
///
/// # Arguments
///
/// * `direction` - Whether this is an onboard (read) or offload (write)
/// * `kind` - Type of remote storage (Object or Disk)
/// * `descriptors` - Remote block descriptors with keys and sizes
/// * `local_blocks` - Local host blocks (source for offload, destination for onboard)
/// * `ctx` - Remote transfer context
/// * `cancel_token` - Cancellation token for cooperative cancellation
///
/// # Returns
///
/// `Ok(())` on success, or `TransferError` on failure/cancellation.
pub async fn execute_remote_transfer<LB>(
    direction: RemoteTransferDirection,
    kind: RemoteStorageKind,
    descriptors: &[RemoteBlockDescriptor],
    local_blocks: &[LB],
    ctx: &RemoteTransferContext,
    cancel_token: &CancellationToken,
) -> Result<(), TransferError>
where
    LB: ReadableBlock + WritableBlock + Local,
    <LB as StorageTypeProvider>::StorageType: NixlDescriptor,
{
    if descriptors.is_empty() || local_blocks.is_empty() {
        return Ok(());
    }

    if descriptors.len() != local_blocks.len() {
        return Err(TransferError::CountMismatch(
            descriptors.len(),
            local_blocks.len(),
        ));
    }

    // Check for early cancellation
    if cancel_token.is_cancelled() {
        return Err(TransferError::Cancelled);
    }

    let nixl_agent_arc = ctx.nixl_agent();
    let agent = nixl_agent_arc
        .as_ref()
        .as_ref()
        .ok_or_else(|| TransferError::ExecutionError("NIXL agent not available".to_string()))?;

    let num_blocks = descriptors.len();

    // Get block size from first local block
    let first_block = &local_blocks[0];
    let block_size = first_block.block_data().block_view()?.size();

    tracing::debug!(
        "Remote transfer: {} blocks, direction={:?}, kind={:?}, block_size={}",
        num_blocks,
        direction,
        kind,
        block_size
    );

    match kind {
        RemoteStorageKind::Object => {
            execute_object_transfer(
                agent,
                direction,
                descriptors,
                local_blocks,
                block_size,
                ctx,
                cancel_token,
            )
            .await
        }
        RemoteStorageKind::Disk => {
            execute_disk_transfer(
                agent,
                direction,
                descriptors,
                local_blocks,
                block_size,
                ctx,
                cancel_token,
            )
            .await
        }
    }
}

/// Execute object storage transfer.
async fn execute_object_transfer<LB>(
    agent: &NixlAgent,
    direction: RemoteTransferDirection,
    descriptors: &[RemoteBlockDescriptor],
    local_blocks: &[LB],
    block_size: usize,
    ctx: &RemoteTransferContext,
    cancel_token: &CancellationToken,
) -> Result<(), TransferError>
where
    LB: ReadableBlock + WritableBlock + Local,
    <LB as StorageTypeProvider>::StorageType: NixlDescriptor,
{
    let num_blocks = descriptors.len();
    let _bucket_template = ctx.bucket_template().unwrap_or("default");

    // Use a scope block to ensure all non-Send types are dropped before await
    let (xfer_req, still_pending) = {
        // Register ALL object storage regions with NIXL
        let mut obj_storages = Vec::with_capacity(num_blocks);
        let mut _registration_handles = Vec::with_capacity(num_blocks);

        // TODO: Add support for string-based object keys via metadata in nixl-sys Rust bindings.
        // For now, we pass the sequence hash (u64) directly as device_id.

        for desc in descriptors.iter() {
            let bucket = match desc.key() {
                RemoteKey::Object(obj_key) => obj_key.bucket.as_str(),
                _ => {
                    return Err(TransferError::IncompatibleTypes(
                        "Expected Object key for object storage transfer".to_string(),
                    ));
                }
            };

            // Use sequence hash directly as device_id - NIXL uses this as the object key
            let object_key = desc.sequence_hash().ok_or_else(|| {
                TransferError::ExecutionError(format!(
                    "Descriptor missing sequence_hash: {:?}",
                    desc.key()
                ))
            })?;

            let obj_storage = ObjectStorage::new(bucket, object_key, block_size).map_err(|e| {
                TransferError::ExecutionError(format!("Failed to create ObjectStorage: {:?}", e))
            })?;

            let handle = agent.register_memory(&obj_storage, None).map_err(|e| {
                TransferError::ExecutionError(format!("Failed to register object storage: {:?}", e))
            })?;

            obj_storages.push(obj_storage);
            _registration_handles.push(handle);
        }

        // Build transfer descriptor lists
        let mut src_dl = XferDescList::new(MemType::Dram).map_err(|e| {
            TransferError::ExecutionError(format!("Failed to create src_dl: {:?}", e))
        })?;
        let mut dst_dl = XferDescList::new(MemType::Object).map_err(|e| {
            TransferError::ExecutionError(format!("Failed to create dst_dl: {:?}", e))
        })?;

        for (block, desc) in local_blocks.iter().zip(descriptors.iter()) {
            let block_view = block.block_data().block_view()?;
            let addr = unsafe { block_view.as_ptr() as usize };

            src_dl.add_desc(addr, block_size, 0);
            dst_dl.add_desc(0, block_size, desc.sequence_hash().unwrap());
        }

        // Determine the transfer operation
        let xfer_op = match direction {
            RemoteTransferDirection::Offload => XferOp::Write,
            RemoteTransferDirection::Onboard => XferOp::Read,
        };

        // Create transfer request
        let agent_name = agent.name();
        let xfer_req = agent
            .create_xfer_req(xfer_op, &src_dl, &dst_dl, &agent_name, None)
            .map_err(|e| {
                TransferError::ExecutionError(format!("Failed to create xfer_req: {:?}", e))
            })?;

        let still_pending = agent.post_xfer_req(&xfer_req, None).map_err(|e| {
            TransferError::ExecutionError(format!("Failed to post xfer_req: {:?}", e))
        })?;

        (xfer_req, still_pending)
    };

    if still_pending {
        let nixl_otel_name = match direction {
            RemoteTransferDirection::Onboard => "kvbm.nixl_read",
            RemoteTransferDirection::Offload => "kvbm.nixl_write",
        };
        let nixl_span = tracing::info_span!(
            "nixl_io",
            otel.name = nixl_otel_name,
            description = "NIXL object store I/O (post + completion wait)",
            num_blocks,
            direction = ?direction,
        );

        let registered = {
            let _enter = nixl_span.enter();
            ctx.register_nixl_transfer(agent, xfer_req)
        };

        use tracing::Instrument;
        match registered {
            Ok(notification) => {
                async {
                    tokio::select! {
                        result = notification => {
                            result.map_err(|e| TransferError::ExecutionError(e.to_string()))
                        }
                        _ = cancel_token.cancelled() => {
                            Err(TransferError::Cancelled)
                        }
                    }
                }.instrument(nixl_span.clone()).await?;
            }
            Err((_, xfer_req)) => {
                poll_transfer_completion_inline(agent, &xfer_req, cancel_token)
                    .instrument(nixl_span)
                    .await?;
            }
        }
    }

    tracing::debug!(
        "Object transfer complete: {} blocks, direction={:?}",
        num_blocks,
        direction
    );

    Ok(())
}

/// Execute disk storage transfer.
async fn execute_disk_transfer<LB>(
    agent: &NixlAgent,
    direction: RemoteTransferDirection,
    descriptors: &[RemoteBlockDescriptor],
    local_blocks: &[LB],
    block_size: usize,
    ctx: &RemoteTransferContext,
    cancel_token: &CancellationToken,
) -> Result<(), TransferError>
where
    LB: ReadableBlock + WritableBlock + Local,
    <LB as StorageTypeProvider>::StorageType: NixlDescriptor,
{
    let num_blocks = descriptors.len();
    let op = if matches!(direction, RemoteTransferDirection::Offload) {
        "write"
    } else {
        "read"
    };
    let base = ctx.base_path().unwrap_or("(none)");
    tracing::info!(
        target: "kvbm-diag",
        direction = op,
        base_path = base,
        num_blocks,
        block_size,
        "Disk transfer starting"
    );

    // For Offload (write): create files
    // For Onboard (read): open existing files
    let create_files = matches!(direction, RemoteTransferDirection::Offload);

    // Determine per-direction backend from transfer flags.
    //
    // Offload: use GDS_MT when DISK_FLAG_GDS_WRITE is set, else POSIX.
    // Onboard: use GDS_MT when DISK_FLAG_GDS_READ is set AND the backend is
    //          available; fall back to POSIX otherwise.
    use crate::block_manager::config::{DISK_FLAG_GDS_READ, DISK_FLAG_GDS_WRITE};
    let flags = ctx.disk_transfer_flags();
    let gds_write = flags & DISK_FLAG_GDS_WRITE != 0;
    let gds_read = flags & DISK_FLAG_GDS_READ != 0;

    // Resolve the GDS_MT backend handle once (None if not loaded in agent).
    let gds_backend = agent.get_backend("GDS_MT");

    // Optional: allow POSIX backend to also open files with O_DIRECT.
    // This is independent from backend selection (GDS_MT vs POSIX).
    let posix_odirect = std::env::var(REMOTE_DISK_O_DIRECT_KEY)
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);

    // Determine whether this direction will use GDS backend.
    let use_gds_backend = if create_files {
        gds_write
    } else {
        gds_read && gds_backend.is_some()
    };

    // Enable O_DIRECT when using GDS backend, and optionally for POSIX when
    // DYN_KVBM_REMOTE_DISK_O_DIRECT is set.
    let use_odirect = use_gds_backend || (!use_gds_backend && posix_odirect);

    let alignment_cfg = *REMOTE_DISK_ALIGNMENT_CONFIG;
    if use_odirect && alignment_cfg.validate && !block_size.is_multiple_of(alignment_cfg.quantum) {
        return Err(TransferError::ExecutionError(format!(
            "O_DIRECT alignment validation failed: block_size must be {}-byte aligned (got {}). \
             Set {}=false to disable validation.",
            alignment_cfg.quantum, block_size, REMOTE_DISK_ALIGNMENT_VALIDATE_KEY
        )));
    }

    // Use a scope block to ensure all non-Send types are dropped before await
    // (OptArgs contains NonNull which is !Send)
    let (xfer_req, still_pending, _disk_storages) = {
        let worker_id = ctx.worker_id() as usize;
        let world_size = ctx.world_size();

        let mut file_paths: Vec<String> = Vec::with_capacity(num_blocks);
        for desc in descriptors.iter() {
            let file_path = match desc.key() {
                RemoteKey::Disk(disk_key) => {
                    let hash = desc.sequence_hash().ok_or_else(|| {
                        TransferError::ExecutionError(
                            "Disk descriptor missing sequence_hash metadata".to_string(),
                        )
                    })?;
                    let base = ctx.base_path().unwrap_or(&disk_key.path);
                    format!("{}/{:016x}_{}_{}", base, hash, worker_id, world_size)
                }
                _ => {
                    return Err(TransferError::IncompatibleTypes(
                        "Expected Disk key for disk storage transfer".to_string(),
                    ));
                }
            };
            file_paths.push(file_path);
        }

        if let Some(first) = file_paths.first() {
            tracing::info!(
                target: "kvbm-diag",
                direction = op,
                first_file = %first,
                num_blocks,
                "Disk transfer files (showing first)"
            );
        }

        let disk_storages = {
            let mut storages: Vec<Arc<SyncMutex<RemoteDiskStorage>>> =
                Vec::with_capacity(num_blocks);
            for path in &file_paths {
                let storage = get_or_open_remote_disk_storage(
                    agent,
                    path,
                    block_size,
                    create_files,
                    use_odirect,
                    use_gds_backend,
                )
                .await?;
                storages.push(storage);
            }
            storages
        };

        // Build transfer descriptor lists for disk
        let mut src_dl = XferDescList::new(MemType::Dram).map_err(|e| {
            TransferError::ExecutionError(format!("Failed to create src_dl: {:?}", e))
        })?;
        let mut dst_dl = XferDescList::new(MemType::File).map_err(|e| {
            TransferError::ExecutionError(format!("Failed to create dst_dl: {:?}", e))
        })?;

        for (block, disk_storage) in local_blocks.iter().zip(disk_storages.iter()) {
            let block_view = block.block_data().block_view()?;
            let addr = unsafe { block_view.as_ptr() as usize };

            if use_odirect && alignment_cfg.validate && !addr.is_multiple_of(alignment_cfg.quantum)
            {
                return Err(TransferError::ExecutionError(format!(
                    "O_DIRECT alignment validation failed: host buffer address must be {}-byte aligned; got 0x{:x}. \
                     Set {}=false to disable validation.",
                    alignment_cfg.quantum, addr, REMOTE_DISK_ALIGNMENT_VALIDATE_KEY
                )));
            }

            // Add DRAM source descriptor
            let _ = src_dl.add_desc(addr, block_size, 0);

            // Add FILE destination descriptor using the actual file descriptor
            let fd = disk_storage.lock().fd();
            let _ = dst_dl.add_desc(0, block_size, fd);
        }

        // Determine the transfer operation
        let xfer_op = match direction {
            RemoteTransferDirection::Offload => XferOp::Write,
            RemoteTransferDirection::Onboard => XferOp::Read,
        };

        // Build OptArgs inside scope block so it's dropped before any await.
        // OptArgs contains NonNull which is !Send; it must not be held across await.
        // Offload: always POSIX when gds_write=false; GDS_MT when true.
        // Onboard: GDS_MT if available, else let NIXL fall through to POSIX.
        let opt_args: Option<OptArgs> = {
            let backend_name = if create_files {
                if gds_write {
                    Some("GDS_MT")
                } else {
                    Some("POSIX")
                }
            } else if gds_read {
                if gds_backend.is_some() {
                    Some("GDS_MT")
                } else {
                    Some("POSIX")
                }
            } else {
                Some("POSIX")
            };
            backend_name.and_then(|name| {
                let backend = agent.get_backend(name)?;
                let mut opt = OptArgs::new().ok()?;
                opt.add_backend(&backend).ok()?;
                Some(opt)
            })
        };
        let opt_ref = opt_args.as_ref();

        // Create transfer request, pinning the backend via OptArgs when set.
        let agent_name = agent.name();
        let xfer_req = agent
            .create_xfer_req(xfer_op, &src_dl, &dst_dl, &agent_name, opt_ref)
            .map_err(|e| {
                TransferError::ExecutionError(format!("Failed to create xfer_req: {:?}", e))
            })?;

        let still_pending = agent.post_xfer_req(&xfer_req, opt_ref).map_err(|e| {
            TransferError::ExecutionError(format!("Failed to post xfer_req: {:?}", e))
        })?;

        (xfer_req, still_pending, disk_storages)
    };

    if still_pending {
        let nixl_otel_name = match direction {
            RemoteTransferDirection::Onboard => "kvbm.nixl_read",
            RemoteTransferDirection::Offload => "kvbm.nixl_write",
        };
        let nixl_span = tracing::info_span!(
            "nixl_io",
            otel.name = nixl_otel_name,
            description = "NIXL disk I/O (post + completion wait)",
            num_blocks,
            direction = op,
        );

        let registered = {
            let _enter = nixl_span.enter();
            ctx.register_nixl_transfer(agent, xfer_req)
        };

        use tracing::Instrument;
        match registered {
            Ok(notification) => {
                async {
                    tokio::select! {
                        result = notification => {
                            result.map_err(|e| TransferError::ExecutionError(e.to_string()))
                        }
                        _ = cancel_token.cancelled() => {
                            Err(TransferError::Cancelled)
                        }
                    }
                }.instrument(nixl_span.clone()).await?;
            }
            Err((_, xfer_req)) => {
                poll_transfer_completion_inline(agent, &xfer_req, cancel_token)
                    .instrument(nixl_span)
                    .await?;
            }
        }
    }

    tracing::debug!(
        "Disk transfer complete: {} blocks, direction={:?}",
        num_blocks,
        direction
    );

    Ok(())
}

#[cfg(all(test, feature = "testing-cuda", feature = "testing-nixl"))]
mod tests {
    use super::*;
    use crate::block_manager::block::transfer::context::TransferContext;
    use crate::block_manager::{
        LayoutConfig,
        block::{BasicMetadata, Block, BlockData, locality},
        config::{RemoteStorageConfig, RemoteTransferContext, DISK_FLAGS_POSIX_BOTH},
        layout::{BlockLayoutConfig, FullyContiguous, nixl::NixlLayout},
        storage::{PinnedAllocator, PinnedStorage},
    };
    use cudarc::driver::CudaContext;
    use std::sync::Arc;

    // Shared NIXL agent with OBJ and POSIX backends
    lazy_static::lazy_static! {
        static ref TEST_AGENT: Arc<Option<NixlAgent>> = {
            let agent = NixlAgent::new("nixl-transfer-test").expect("Failed to create NIXL agent");

            // Create OBJ backend for object storage
            if let Ok((_, params)) = agent.get_plugin_params("OBJ") {
                match agent.create_backend("OBJ", &params) {
                    Ok(_) => eprintln!("OBJ backend created"),
                    Err(e) => eprintln!("OBJ backend failed: {}", e),
                }
            } else {
                eprintln!("OBJ plugin not found");
            }

            // Create POSIX backend for disk storage
            if let Ok((_, params)) = agent.get_plugin_params("POSIX") {
                match agent.create_backend("POSIX", &params) {
                    Ok(_) => eprintln!("POSIX backend created"),
                    Err(e) => eprintln!("POSIX backend failed: {}", e),
                }
            } else {
                eprintln!("POSIX plugin not found");
            }

            Arc::new(Some(agent))
        };

        static ref CUDA_CTX: Arc<CudaContext> = {
            CudaContext::new(0).expect("Failed to create CUDA context")
        };
    }

    fn create_test_layout(num_blocks: usize) -> FullyContiguous<PinnedStorage> {
        let config = LayoutConfig::builder()
            .num_blocks(num_blocks)
            .num_layers(2)
            .outer_dim(1)
            .page_size(4)
            .inner_dim(64)
            .dtype_width_bytes(2)
            .build()
            .unwrap();

        let allocator = PinnedAllocator::new().unwrap();
        FullyContiguous::allocate(config, &allocator).unwrap()
    }

    fn create_transfer_context() -> Arc<TransferContext> {
        let stream = CUDA_CTX.default_stream();
        let handle = tokio::runtime::Handle::current();
        Arc::new(
            TransferContext::new(TEST_AGENT.clone(), stream, handle, None)
                .expect("TransferContext::new for NIXL tests"),
        )
    }

    fn create_disk_remote_context(base: Arc<TransferContext>, path: &str) -> RemoteTransferContext {
        RemoteTransferContext::new(base, RemoteStorageConfig::disk(path, DISK_FLAGS_POSIX_BOTH))
    }

    fn create_object_remote_context(
        base: Arc<TransferContext>,
        bucket: &str,
    ) -> RemoteTransferContext {
        RemoteTransferContext::new(base, RemoteStorageConfig::object(bucket))
    }

    #[tokio::test]
    async fn test_execute_remote_transfer_empty_descriptors() {
        let base_ctx = create_transfer_context();
        let remote_ctx = create_disk_remote_context(base_ctx, "/tmp/nixl-test");
        let cancel_token = CancellationToken::new();

        let descriptors: Vec<RemoteBlockDescriptor> = vec![];
        let local_blocks: Vec<Block<PinnedStorage, locality::Local, BasicMetadata>> = vec![];

        let result = execute_remote_transfer(
            RemoteTransferDirection::Offload,
            RemoteStorageKind::Disk,
            &descriptors,
            &local_blocks,
            &remote_ctx,
            &cancel_token,
        )
        .await;

        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_execute_remote_transfer_count_mismatch() {
        let base_ctx = create_transfer_context();
        let remote_ctx = create_disk_remote_context(base_ctx, "/tmp/nixl-test");
        let cancel_token = CancellationToken::new();

        // Create 2 descriptors but 1 block - should fail
        let descriptors = vec![
            RemoteBlockDescriptor::disk_from_hash("/tmp", 0x1234, 1024, 0, 1),
            RemoteBlockDescriptor::disk_from_hash("/tmp", 0x5678, 1024, 0, 1),
        ];

        let mut layout = create_test_layout(1);
        layout
            .nixl_register(TEST_AGENT.as_ref().as_ref().unwrap(), None)
            .unwrap();
        let layout = Arc::new(layout);
        let blocks: Vec<Block<PinnedStorage, locality::Local, BasicMetadata>> = (0..1)
            .map(|i| {
                let data = BlockData::new(layout.clone(), i, 0, 0);
                Block::new(data, BasicMetadata::default()).unwrap()
            })
            .collect();

        let result = execute_remote_transfer(
            RemoteTransferDirection::Offload,
            RemoteStorageKind::Disk,
            &descriptors,
            &blocks,
            &remote_ctx,
            &cancel_token,
        )
        .await;

        assert!(matches!(result, Err(TransferError::CountMismatch(2, 1))));
    }

    #[tokio::test]
    async fn test_execute_remote_transfer_early_cancellation() {
        let base_ctx = create_transfer_context();
        let remote_ctx = create_disk_remote_context(base_ctx, "/tmp/nixl-test");
        let cancel_token = CancellationToken::new();

        // Cancel before starting
        cancel_token.cancel();

        let descriptors = vec![RemoteBlockDescriptor::disk_from_hash(
            "/tmp", 0x1234, 1024, 0, 1,
        )];

        let mut layout = create_test_layout(1);
        layout
            .nixl_register(TEST_AGENT.as_ref().as_ref().unwrap(), None)
            .unwrap();
        let layout = Arc::new(layout);
        let blocks: Vec<Block<PinnedStorage, locality::Local, BasicMetadata>> = (0..1)
            .map(|i| {
                let data = BlockData::new(layout.clone(), i, 0, 0);
                Block::new(data, BasicMetadata::default()).unwrap()
            })
            .collect();

        let result = execute_remote_transfer(
            RemoteTransferDirection::Offload,
            RemoteStorageKind::Disk,
            &descriptors,
            &blocks,
            &remote_ctx,
            &cancel_token,
        )
        .await;

        assert!(matches!(result, Err(TransferError::Cancelled)));
    }

    /// Test disk transfer using POSIX backend - writes data to disk and reads it back
    #[tokio::test]
    async fn test_posix_disk_transfer_roundtrip() {
        use tempfile::tempdir;

        let temp_dir = tempdir().unwrap();
        let base_path = temp_dir.path().to_str().unwrap().to_string();

        let base_ctx = create_transfer_context();
        let remote_ctx = create_disk_remote_context(base_ctx, &base_path);
        let cancel_token = CancellationToken::new();

        // Create blocks and fill with test data
        let mut layout = create_test_layout(2);
        layout
            .nixl_register(TEST_AGENT.as_ref().as_ref().unwrap(), None)
            .unwrap();
        let block_size = layout.layout_data_bytes() / layout.num_blocks();
        let layout = Arc::new(layout);

        let mut blocks: Vec<Block<PinnedStorage, locality::Local, BasicMetadata>> = (0..2)
            .map(|i| {
                let data = BlockData::new(layout.clone(), i, 0, 0);
                Block::new(data, BasicMetadata::default()).unwrap()
            })
            .collect();

        // Fill blocks with recognizable pattern
        for (i, block) in blocks.iter_mut().enumerate() {
            let mut view = block.block_data_mut().block_view_mut().unwrap();
            let slice = unsafe { std::slice::from_raw_parts_mut(view.as_mut_ptr(), view.size()) };
            for (j, byte) in slice.iter_mut().enumerate() {
                *byte = ((i * 100 + j) % 256) as u8;
            }
        }

        let descriptors: Vec<RemoteBlockDescriptor> = (0..2u64)
            .map(|i| {
                RemoteBlockDescriptor::disk_from_hash(&base_path, 0x1000 + i, block_size, 0, 1)
            })
            .collect();

        // Offload (write to disk via POSIX)
        let result = execute_remote_transfer(
            RemoteTransferDirection::Offload,
            RemoteStorageKind::Disk,
            &descriptors,
            &blocks,
            &remote_ctx,
            &cancel_token,
        )
        .await;

        if result.is_err() {
            eprintln!(
                "POSIX disk transfer test skipped - backend may not be available: {:?}",
                result
            );
            return;
        }
        assert!(result.is_ok(), "POSIX Offload failed: {:?}", result);

        // Drop offload blocks to ensure we're not reusing cached memory
        drop(blocks);

        // Create fresh blocks for onboarding (different memory)
        let mut onboard_layout = create_test_layout(2);
        onboard_layout
            .nixl_register(TEST_AGENT.as_ref().as_ref().unwrap(), None)
            .unwrap();
        let onboard_layout = Arc::new(onboard_layout);

        let onboard_blocks: Vec<Block<PinnedStorage, locality::Local, BasicMetadata>> = (0..2)
            .map(|i| {
                let data = BlockData::new(onboard_layout.clone(), i, 0, 0);
                Block::new(data, BasicMetadata::default()).unwrap()
            })
            .collect();

        // Verify onboard blocks start zeroed (not containing our pattern)
        for block in onboard_blocks.iter() {
            let view = block.block_data().block_view().unwrap();
            let slice = unsafe { std::slice::from_raw_parts(view.as_ptr(), view.size()) };
            assert!(
                slice.iter().all(|&b| b == 0),
                "Onboard blocks should start zeroed"
            );
        }

        // Onboard (read from disk via POSIX) into fresh blocks
        let result = execute_remote_transfer(
            RemoteTransferDirection::Onboard,
            RemoteStorageKind::Disk,
            &descriptors,
            &onboard_blocks,
            &remote_ctx,
            &cancel_token,
        )
        .await;

        assert!(result.is_ok(), "POSIX Onboard failed: {:?}", result);

        // Verify data was restored to the new blocks
        for (i, block) in onboard_blocks.iter().enumerate() {
            let view = block.block_data().block_view().unwrap();
            let slice = unsafe { std::slice::from_raw_parts(view.as_ptr(), view.size()) };
            for (j, &byte) in slice.iter().enumerate() {
                let expected = ((i * 100 + j) % 256) as u8;
                assert_eq!(
                    byte, expected,
                    "POSIX: Data mismatch at block {} byte {}",
                    i, j
                );
            }
        }
        eprintln!("POSIX disk roundtrip transfer successful (using separate blocks for onboard)");
    }

    /// Test object storage transfer using OBJ backend - writes data to S3/object store and reads it back
    #[tokio::test]
    async fn test_obj_object_storage_transfer_roundtrip() {
        let bucket =
            std::env::var("NIXL_TEST_BUCKET").unwrap_or_else(|_| "nixl-test-bucket".to_string());

        let base_ctx = create_transfer_context();
        let remote_ctx = create_object_remote_context(base_ctx, &bucket);
        let cancel_token = CancellationToken::new();

        // Create blocks and fill with test data
        let mut layout = create_test_layout(2);
        layout
            .nixl_register(TEST_AGENT.as_ref().as_ref().unwrap(), None)
            .unwrap();
        let block_size = layout.layout_data_bytes() / layout.num_blocks();
        let layout = Arc::new(layout);

        let mut blocks: Vec<Block<PinnedStorage, locality::Local, BasicMetadata>> = (0..2)
            .map(|i| {
                let data = BlockData::new(layout.clone(), i, 0, 0);
                Block::new(data, BasicMetadata::default()).unwrap()
            })
            .collect();

        for (i, block) in blocks.iter_mut().enumerate() {
            let mut view = block.block_data_mut().block_view_mut().unwrap();
            let slice = unsafe { std::slice::from_raw_parts_mut(view.as_mut_ptr(), view.size()) };
            for (j, byte) in slice.iter_mut().enumerate() {
                *byte = ((i * 200 + j) % 256) as u8;
            }
        }

        // Use unique sequence hashes for this test run
        let test_id = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;

        let descriptors: Vec<RemoteBlockDescriptor> = (0..2u64)
            .map(|i| RemoteBlockDescriptor::object_from_hash(&bucket, test_id + i, block_size))
            .collect();

        // Offload (write to object storage via OBJ)
        let result = execute_remote_transfer(
            RemoteTransferDirection::Offload,
            RemoteStorageKind::Object,
            &descriptors,
            &blocks,
            &remote_ctx,
            &cancel_token,
        )
        .await;

        if result.is_err() {
            eprintln!(
                "OBJ object storage test skipped - backend may not be available or S3 not configured: {:?}",
                result
            );
            return;
        }
        assert!(result.is_ok(), "OBJ Offload failed: {:?}", result);

        // Drop offload blocks to ensure we're not reusing cached memory
        drop(blocks);

        // Create fresh blocks for onboarding (different memory)
        let mut onboard_layout = create_test_layout(2);
        onboard_layout
            .nixl_register(TEST_AGENT.as_ref().as_ref().unwrap(), None)
            .unwrap();
        let onboard_layout = Arc::new(onboard_layout);

        let onboard_blocks: Vec<Block<PinnedStorage, locality::Local, BasicMetadata>> = (0..2)
            .map(|i| {
                let data = BlockData::new(onboard_layout.clone(), i, 0, 0);
                Block::new(data, BasicMetadata::default()).unwrap()
            })
            .collect();

        // Verify onboard blocks start zeroed (not containing our pattern)
        for block in onboard_blocks.iter() {
            let view = block.block_data().block_view().unwrap();
            let slice = unsafe { std::slice::from_raw_parts(view.as_ptr(), view.size()) };
            assert!(
                slice.iter().all(|&b| b == 0),
                "Onboard blocks should start zeroed"
            );
        }

        let result = execute_remote_transfer(
            RemoteTransferDirection::Onboard,
            RemoteStorageKind::Object,
            &descriptors,
            &onboard_blocks,
            &remote_ctx,
            &cancel_token,
        )
        .await;

        assert!(result.is_ok(), "OBJ Onboard failed: {:?}", result);

        // Verify data was restored to the new blocks
        for (i, block) in onboard_blocks.iter().enumerate() {
            let view = block.block_data().block_view().unwrap();
            let slice = unsafe { std::slice::from_raw_parts(view.as_ptr(), view.size()) };
            for (j, &byte) in slice.iter().enumerate() {
                let expected = ((i * 200 + j) % 256) as u8;
                assert_eq!(
                    byte, expected,
                    "OBJ: Data mismatch at block {} byte {}",
                    i, j
                );
            }
        }
        eprintln!("OBJ object storage roundtrip transfer successful");
    }
}
