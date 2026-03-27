// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_llm::block_manager::connector::protocol::{
    RequestType, SlotKey, TransferType, WorkerTransferRequest,
};
use dynamo_llm::block_manager::connector::scheduler::{
    Scheduler, TransferSchedulerClient, WorkerSchedulerClient,
};

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, OnceLock};

use super::*;
use crate::block_manager::distributed::{get_leader_zmq_ack_url, get_leader_zmq_pub_url};
use crate::{block_manager::distributed::VllmTensor, to_pyerr};

use crate::block_manager::distributed::PyLayoutType;
use crate::{
    extract_distributed_runtime_from_obj, get_current_cancel_token, get_current_tokio_handle,
};
use anyhow;
use dynamo_llm::block_manager::distributed::{KvbmWorker, KvbmWorkerConfig};
use dynamo_llm::block_manager::layout::LayoutType;
use dynamo_llm::block_manager::storage::torch::TorchTensor;
use dynamo_runtime::DistributedRuntime;
use dynamo_runtime::utils::task::CriticalTaskExecutionHandle;

pub trait Worker: Send + Sync {
    #[allow(clippy::too_many_arguments)]
    fn register_kv_caches(
        &mut self,
        num_device_blocks: usize,
        page_size: usize,
        device_id: usize,
        dtype_width_bytes: usize,
        kv_caches: Vec<(String, Arc<VllmTensor>)>,
        raw_event_handles: Vec<u64>,
        device_layout_type: Option<LayoutType>,
        host_layout_type: Option<LayoutType>,
        disk_layout_type: Option<LayoutType>,
    ) -> anyhow::Result<()>;

    fn bind_connector_metadata(&mut self, metadata: Vec<u8>) -> anyhow::Result<()>;

    fn clear_connector_metadata(&mut self);

    fn save_kv_layer(&mut self, layer_name: String) -> anyhow::Result<()>;

    fn get_finished(
        &mut self,
        finished_requests: HashSet<String>,
    ) -> (HashSet<String>, HashSet<String>);

    /// Get block IDs that failed to load and clear the set
    fn get_block_ids_with_load_errors(&mut self) -> HashSet<u32>;

    /// v0.18 worker metadata: JSON `{"onboard":{req:[uuid..]},"offload":{...},"failed":{...}}`.
    fn build_connector_worker_meta_json(&mut self) -> Option<String> {
        None
    }
}

pub struct KvConnectorWorker {
    _drt: Option<Arc<DistributedRuntime>>,
    kvbm_worker: OnceLock<KvbmWorker>,
    connector: WorkerSchedulerClient,
    transfer_client: TransferSchedulerClient,

    kv_cache_layers: Vec<(String, Arc<dyn TorchTensor>)>,

    /// Completion snapshots from the leader (refreshed each iteration via metadata).
    loads_done: HashSet<String>,
    stores_done: HashSet<String>,
    failed_requests: HashSet<String>,

    /// Current active epoch per request id, used only as the external adapter.
    active_epochs: HashMap<String, SlotKey>,
    /// Request IDs already reported as onboarding-complete (dedup).
    reported_onboarding: HashSet<String>,

    /// For now, offloading operations will be enqueued at the end of the forward pass
    offloading_operations: Vec<WorkerTransferRequest>,

    bound: bool,
    iteration: u64,
    layers_complete: usize,

    /// cuda events created by the python side
    layer_events: Vec<u64>,

    /// Map epoch key to (uuid → block_ids) for error tracking (Load operations only)
    request_to_blocks: HashMap<SlotKey, HashMap<uuid::Uuid, Vec<usize>>>,

    /// Block IDs that failed to load.
    /// Uses u32 since vLLM block IDs are 32-bit. Protocol uses usize for flexibility,
    /// but actual block counts won't exceed u32::MAX in practice.
    failed_block_ids: HashSet<u32>,

    /// Pending failure notifications not yet processed (epoch → failed UUIDs)
    pending_failures: HashMap<SlotKey, HashSet<uuid::Uuid>>,

    /// Onboarding epochs that the scheduler has declared failed and whose load
    /// errors should be surfaced back to vLLM.
    failed_onboarding_keys: HashSet<SlotKey>,
}

impl KvConnectorWorker {
    fn track_epoch(&mut self, key: &SlotKey) {
        self.active_epochs
            .insert(key.request_id.clone(), key.clone());
    }

    fn new(drt: Option<Arc<DistributedRuntime>>, vllm_worker_id: String) -> anyhow::Result<Self> {
        let runtime = get_current_tokio_handle();

        let (scheduler, worker_client, transfer_client) =
            Scheduler::new(vllm_worker_id.clone(), get_current_cancel_token());

        CriticalTaskExecutionHandle::new_with_runtime(
            move |_| {
                let mut scheduler = scheduler;
                async move { scheduler.run().await }
            },
            get_current_cancel_token(),
            "kv-connector-scheduler-task",
            &runtime,
        )?
        .detach();

        tracing::info!(
            "KvConnectorWorker initialized with worker_id: {}",
            vllm_worker_id
        );

        Ok(Self {
            _drt: drt,
            kvbm_worker: OnceLock::new(),
            connector: worker_client,
            transfer_client,
            loads_done: HashSet::new(),
            stores_done: HashSet::new(),
            failed_requests: HashSet::new(),
            active_epochs: HashMap::new(),
            reported_onboarding: HashSet::new(),
            offloading_operations: Vec::new(),
            bound: false,
            iteration: 0,
            layers_complete: 0,
            kv_cache_layers: Vec::new(),
            layer_events: Vec::new(),
            request_to_blocks: HashMap::new(),
            failed_block_ids: HashSet::new(),
            pending_failures: HashMap::new(),
            failed_onboarding_keys: HashSet::new(),
        })
    }

}

impl Worker for KvConnectorWorker {
    /// Registers the KV caches with the KVBM worker.
    ///
    /// The Dynamo KVBM worker is lazily initialized when the first KV cache is registered.
    /// This process establishes a connection between all KVBM workers and the leader.
    fn register_kv_caches(
        &mut self,
        num_device_blocks: usize,
        page_size: usize,
        device_id: usize,
        dtype_width_bytes: usize,
        kv_caches: Vec<(String, Arc<VllmTensor>)>,
        raw_event_handles: Vec<u64>,
        device_layout_type: Option<LayoutType>,
        host_layout_type: Option<LayoutType>,
        disk_layout_type: Option<LayoutType>,
    ) -> anyhow::Result<()> {
        if self.kvbm_worker.get().is_some() {
            tracing::warn!("kvbm worker already registered");
            return Err(anyhow::anyhow!("kvbm worker already registered"));
        }

        assert_eq!(
            kv_caches.len(),
            raw_event_handles.len(),
            "kv_caches and raw_event_handles must have the same length"
        );

        // Process kv_caches in layer execution order (already sorted by layer index)
        let mut vllm_tensors = Vec::new();
        let mut first_tensor_shape: Option<Vec<usize>> = None;

        for (layer_name, vllm_tensor) in kv_caches {
            tracing::trace!("Registering KV cache layer: {layer_name}, tensor: {vllm_tensor:?}");

            // Capture the shape of the first tensor for layout detection
            if first_tensor_shape.is_none() {
                first_tensor_shape = Some(vllm_tensor.shape());
            }

            // Store for later lookup by name
            self.kv_cache_layers
                .push((layer_name, vllm_tensor.clone() as Arc<dyn TorchTensor>));

            // Build ordered tensor list for worker config
            vllm_tensors.push(vllm_tensor as Arc<dyn TorchTensor>);
        }

        self.layer_events = raw_event_handles;

        // Auto-detect device layout type if not explicitly provided
        let detected_device_layout_type = match device_layout_type {
            Some(layout) => layout,
            None => {
                if let Some(ref shape) = first_tensor_shape {
                    match LayoutType::layer_separate_auto(shape, num_device_blocks) {
                        Ok(detected) => {
                            tracing::info!(
                                "Auto-detected device layout from tensor shape: {:?}",
                                detected
                            );
                            detected
                        }
                        Err(e) => {
                            tracing::warn!(
                                "Failed to auto-detect layout from shape {:?}: {}. Using default.",
                                shape,
                                e
                            );
                            LayoutType::layer_separate_auto_default()
                        }
                    }
                } else {
                    tracing::warn!("No tensors available for layout detection. Using default.");
                    LayoutType::layer_separate_auto_default()
                }
            }
        };

        let config = KvbmWorkerConfig::builder()
            .cancel_token(get_current_cancel_token())
            .num_device_blocks(num_device_blocks)
            .page_size(page_size)
            .tensors(vllm_tensors)
            .device_id(device_id)
            .dtype_width_bytes(dtype_width_bytes)
            .scheduler_client(Some(self.transfer_client.clone()))
            .device_layout_type(detected_device_layout_type)
            .host_layout_type(host_layout_type.unwrap_or(LayoutType::FullyContiguous))
            .disk_layout_type(disk_layout_type.unwrap_or(LayoutType::FullyContiguous))
            .leader_pub_url(get_leader_zmq_pub_url())
            .leader_ack_url(get_leader_zmq_ack_url())
            .build()?;

        let worker = get_current_tokio_handle().block_on(async move {
            let worker = KvbmWorker::new(config, false).await?;
            anyhow::Ok(worker)
        })?;

        self.kvbm_worker
            .set(worker)
            .map_err(|_| anyhow::anyhow!("failed to set kvbm worker"))?;

        Ok(())
    }

    /// Loads the metadata from the leader.
    /// This action translates the metadata into a set of actions that the worker will perform.
    /// All actions much be assigned to a slot before [`KvConnectorWorker::clear_metadata`] is called.
    fn bind_connector_metadata(&mut self, metadata: Vec<u8>) -> anyhow::Result<()> {
        // debug_assert!(!self.bound, "connector metadata already bound");
        let metadata: ConnectorMetadata = serde_json::from_slice(&metadata)?;
        self.loads_done = metadata.loads_done.clone();
        self.stores_done = metadata.stores_done.clone();
        self.failed_requests = metadata.failed.clone();
        // Onboarding cardinality must be derived from the concrete immediate
        // load ops the worker is about to enqueue, not trusted from metadata
        // blindly. `NewSlotInfo.expected_immediate_ops` is treated as a
        // checksum only.
        let derived_immediate_loads: HashMap<SlotKey, u64> = metadata
            .operations
            .iter()
            .filter(|op| {
                op.transfer_type == TransferType::Load
                    && op.request_type == RequestType::Immediate
            })
            .fold(HashMap::new(), |mut counts, op| {
                *counts.entry(op.key.clone()).or_insert(0) += 1;
                counts
            });
        self.bound = true;
        self.iteration = metadata.iteration;
        self.layers_complete = 0;
        tracing::debug!(
            iteration = self.iteration,
            "bound new metadata: {metadata:#?}"
        );

        self.connector.start_next_iteration()?;

        debug_assert_eq!(
            self.connector.iteration(),
            metadata.iteration,
            "iteration mismatch"
        );

        // self.engine_tx
        //     .send(EngineMessage::UpdateIteration(self.iteration))
        //     .map_err(to_pyerr)?;

        // local actions
        // - create a request slot for each new request
        // - for each action in the metadata, add the action to the request slot
        // - send the list of actions to the engine to track completion

        for slot_info in &metadata.new_slots {
            let actual_immediate_ops = derived_immediate_loads
                .get(&slot_info.key)
                .copied()
                .unwrap_or(0);
            match slot_info.kind {
                NewSlotKind::Onboarding => {
                    if actual_immediate_ops == 0 {
                        return Err(anyhow::anyhow!(
                            "invalid onboarding metadata for request '{}': onboarding slot carries zero immediate load ops",
                            slot_info.key.request_id
                        ));
                    }
                    if slot_info.expected_immediate_ops != actual_immediate_ops {
                        return Err(anyhow::anyhow!(
                            "invalid onboarding metadata for request '{}': leader expected {} immediate ops but worker derived {} from operations",
                            slot_info.key.request_id,
                            slot_info.expected_immediate_ops,
                            actual_immediate_ops
                        ));
                    }
                }
                NewSlotKind::Prefill => {
                    if actual_immediate_ops != 0 {
                        return Err(anyhow::anyhow!(
                            "invalid prefill metadata for request '{}': prefill slot carries {} immediate load ops",
                            slot_info.key.request_id,
                            actual_immediate_ops
                        ));
                    }
                    if slot_info.expected_immediate_ops != 0 {
                        return Err(anyhow::anyhow!(
                            "invalid prefill metadata for request '{}': expected_immediate_ops must be 0, got {}",
                            slot_info.key.request_id,
                            slot_info.expected_immediate_ops
                        ));
                    }
                }
            }

            if let Some(existing_key) = self.active_epochs.get(&slot_info.key.request_id).cloned() {
                if existing_key != slot_info.key && self.connector.has_key(&existing_key) {
                    tracing::debug!(
                        request_id = %slot_info.key.request_id,
                        previous_generation = existing_key.generation,
                        new_generation = slot_info.key.generation,
                        slot_kind = ?slot_info.kind,
                        expected_immediate_ops = actual_immediate_ops,
                        "replacing previous epoch slot"
                    );
                    let _ = self.connector.remove_key(&existing_key);
                } else if existing_key == slot_info.key && self.connector.has_key(&existing_key) {
                    continue;
                }
            }

            tracing::debug!(
                request_id = %slot_info.key.request_id,
                slot_kind = ?slot_info.kind,
                expected_immediate_ops = actual_immediate_ops,
                "creating connector slot"
            );
            self.connector.create_slot_with_key_and_immediate_ops(
                slot_info.key.clone(),
                actual_immediate_ops,
            )?;
            self.track_epoch(&slot_info.key);
        }

        let mut onboarding_operations = Vec::new();
        let mut offloading_operations = Vec::new();

        for operation in metadata.operations {
            tracing::debug!(
                request_id = operation.key.request_id, operation_id = %operation.uuid,
                "adding operation to slot: {operation:#?}"
            );

            match operation.transfer_type {
                TransferType::Load => onboarding_operations.push(operation),
                TransferType::Store => offloading_operations.push(operation),
            }
        }

        // immediately enqueue the onboarding operations
        for operation in onboarding_operations {
            let uuid = operation.uuid;

            // Store block_ids per operation UUID for error tracking
            if !operation.block_ids.is_empty() {
                self.request_to_blocks
                    .entry(operation.key.clone())
                    .or_default()
                    .insert(uuid, operation.block_ids.clone());
            }

            let key = operation.key.clone();
            self.connector.enqueue_request(operation)?;
            self.track_epoch(&key);
        }

        self.offloading_operations = offloading_operations;

        Ok(())
    }

    /// Clears the connector metadata and marks the iteration as complete.
    fn clear_connector_metadata(&mut self) {
        tracing::debug!(iteration = self.iteration, "clearing connector metadata");
        debug_assert!(self.bound, "connector metadata not bound");
        self.bound = false;
        self.iteration = 0; // always reset; leader drives the counter
        self.layers_complete = 0;
        self.connector
            .mark_iteration_complete()
            .expect("failed to mark iteration complete");
    }

    /// Trigger layer-wise completion signals.
    /// Trigger block-wise completion signals afer last layer.
    fn save_kv_layer(&mut self, _layer_name: String) -> anyhow::Result<()> {
        self.layers_complete += 1;
        tracing::debug!(
            iteration = self.iteration,
            layers_complete = self.layers_complete,
            total_layers = self.kv_cache_layers.len(),
            pending_offload_ops = self.offloading_operations.len(),
            "save_kv_layer called"
        );
        if self.layers_complete == self.kv_cache_layers.len() {
            let offloading_operations = std::mem::take(&mut self.offloading_operations);

            tracing::trace!(
                iteration = self.iteration,
                num_operations = offloading_operations.len(),
                "All layers complete, enqueuing {} offload operations",
                offloading_operations.len()
            );

            // block on the the completion of the last layer
            // todo(ryan): capture the context, pass this to the scheduler to do the await on another thread
            // or put the event on a stream and use stream waits to keep it all on device.
            if self.layers_complete - 1 < self.layer_events.len() {
                let ev = self.layer_events[self.layers_complete - 1];
                if ev != 0 {
                    event_sync_blocking(ev);
                }
            }
            for operation in &offloading_operations {
                tracing::debug!(
                    request_id = %operation.key.request_id,
                    operation_id = %operation.uuid,
                    "Enqueuing offload operation to scheduler"
                );
                self.connector.enqueue_request(operation.clone())?;
            }
        }
        Ok(())
    }

    fn get_finished(
        &mut self,
        finished_requests: HashSet<String>,
    ) -> (HashSet<String>, HashSet<String>) {
        let mut is_finished_offloading = HashSet::new();
        let mut is_finished_onboarding = HashSet::new();

        for request_id in finished_requests {
            let Some(key) = self.active_epochs.get(&request_id).cloned() else {
                continue;
            };
            let stores_done = self.stores_done.contains(&request_id);
            if stores_done {
                if self.connector.has_key(&key) {
                    let _ = self.connector.remove_key(&key);
                }
                self.active_epochs.remove(&request_id);
                self.reported_onboarding.remove(&request_id);
                is_finished_offloading.insert(request_id);
            }
        }

        let active_req_ids: Vec<_> = self
            .active_epochs
            .keys()
            .filter(|rid| !self.reported_onboarding.contains(*rid))
            .cloned()
            .collect();
        for request_id in active_req_ids {
            if self.loads_done.contains(&request_id) {
                is_finished_onboarding.insert(request_id.clone());
                self.reported_onboarding.insert(request_id.clone());
            }
            if self.failed_requests.contains(&request_id) {
                if let Some(key) = self.active_epochs.get(&request_id) {
                    self.failed_onboarding_keys.insert(key.clone());
                }
            }
        }

        tracing::info!(
            iteration = self.iteration,
            finished_offloading = ?is_finished_offloading,
            finished_onboarding = ?is_finished_onboarding,
            "worker get_finished: returning request IDs to vLLM"
        );

        (is_finished_offloading, is_finished_onboarding)
    }

    fn get_block_ids_with_load_errors(&mut self) -> HashSet<u32> {
        let failed_onboarding_keys: Vec<SlotKey> =
            self.failed_onboarding_keys.iter().cloned().collect();
        for key in &failed_onboarding_keys {
            if let Some(failed_uuids) = self.pending_failures.get(key)
                && let Some(uuid_to_blocks) = self.request_to_blocks.get(key)
            {
                for failed_uuid in failed_uuids {
                    if let Some(block_ids) = uuid_to_blocks.get(failed_uuid) {
                        for &block_id in block_ids {
                            self.failed_block_ids.insert(block_id as u32);
                        }
                    }
                }
            }
        }
        for key in failed_onboarding_keys {
            self.failed_onboarding_keys.remove(&key);
            self.request_to_blocks.remove(&key);
            self.pending_failures.remove(&key);
        }

        std::mem::take(&mut self.failed_block_ids)
    }

    fn build_connector_worker_meta_json(&mut self) -> Option<String> {
        None
    }
}

#[pyclass]
pub struct PyKvConnectorWorker {
    connector_worker: Box<dyn Worker>,
}

#[pymethods]
impl PyKvConnectorWorker {
    #[new]
    #[pyo3(signature = (py_drt, vllm_worker_id))]
    pub fn new(py_drt: Option<PyObject>, vllm_worker_id: String) -> PyResult<Self> {
        let drt: Option<Arc<DistributedRuntime>> = Python::with_gil(|py| {
            if let Some(obj) = py_drt {
                extract_distributed_runtime_from_obj(py, obj)
            } else {
                Ok(None)
            }
        })?;

        let connector_worker: Box<dyn Worker> =
            Box::new(KvConnectorWorker::new(drt, vllm_worker_id).map_err(to_pyerr)?);
        Ok(Self { connector_worker })
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (num_device_blocks, page_size, device_id, dtype_width_bytes, kv_caches, raw_event_handles, device_layout_type=None, host_layout_type=None, disk_layout_type=None))]
    pub fn register_kv_caches(
        &mut self,
        num_device_blocks: usize,
        page_size: usize,
        device_id: usize,
        dtype_width_bytes: usize,
        kv_caches: Vec<(String, Py<PyAny>)>,
        raw_event_handles: Vec<u64>,
        device_layout_type: Option<PyLayoutType>,
        host_layout_type: Option<PyLayoutType>,
        disk_layout_type: Option<PyLayoutType>,
    ) -> PyResult<()> {
        // Convert Python tensors to Rust VllmTensor objects
        let mut rust_kv_caches = Vec::new();
        for (layer_name, py_tensor) in kv_caches {
            let vllm_tensor = Arc::new(VllmTensor::new(py_tensor).map_err(to_pyerr)?);
            rust_kv_caches.push((layer_name, vllm_tensor));
        }

        self.connector_worker
            .register_kv_caches(
                num_device_blocks,
                page_size,
                device_id,
                dtype_width_bytes,
                rust_kv_caches,
                raw_event_handles,
                device_layout_type.map(|py_layout| py_layout.into()),
                host_layout_type.map(|py_layout| py_layout.into()),
                disk_layout_type.map(|py_layout| py_layout.into()),
            )
            .map_err(to_pyerr)
    }

    pub fn bind_connector_metadata(&mut self, metadata: Vec<u8>) -> PyResult<()> {
        self.connector_worker
            .bind_connector_metadata(metadata)
            .map_err(to_pyerr)
    }

    pub fn clear_connector_metadata(&mut self) {
        self.connector_worker.clear_connector_metadata()
    }

    pub fn save_kv_layer(&mut self, layer_name: String, _kv_layer: Py<PyAny>) -> PyResult<()> {
        // Note: kv_layer is not used in the current implementation
        self.connector_worker
            .save_kv_layer(layer_name)
            .map_err(to_pyerr)
    }

    pub fn get_finished(
        &mut self,
        finished_requests: HashSet<String>,
    ) -> (HashSet<String>, HashSet<String>) {
        self.connector_worker.get_finished(finished_requests)
    }

    /// Get block IDs that failed to load and clear the set
    pub fn get_block_ids_with_load_errors(&mut self) -> HashSet<u32> {
        self.connector_worker.get_block_ids_with_load_errors()
    }

    /// Worker meta is no longer needed — all completion signaling goes through TransferSignal.
    pub fn build_connector_worker_meta_json(&mut self) -> Option<String> {
        None
    }
}

use cudarc::driver::sys::{
    CUcontext, CUevent, cuCtxGetCurrent, cuEventSynchronize, cudaError_enum,
};
use std::ptr;

// todo(ryan): we will need this if we farm off the cuEventSynchronize to another thread
fn _get_current_context() -> CUcontext {
    let mut ctx: CUcontext = ptr::null_mut();
    let status = unsafe { cuCtxGetCurrent(&mut ctx) };
    assert_eq!(
        status,
        cudaError_enum::CUDA_SUCCESS,
        "cuCtxGetCurrent failed"
    );
    assert!(!ctx.is_null(), "Torch has not set a CUDA context");
    ctx
}

pub fn event_sync_blocking(event: u64) {
    let status = unsafe { cuEventSynchronize(event as CUevent) };
    assert_eq!(
        status,
        cudaError_enum::CUDA_SUCCESS,
        "cuEventSynchronize failed"
    );
}
