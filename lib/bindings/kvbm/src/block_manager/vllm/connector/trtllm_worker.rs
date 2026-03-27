// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_llm::block_manager::connector::protocol::{SlotKey, TransferType};
use dynamo_llm::block_manager::connector::scheduler::{
    Scheduler, SchedulerMessage, TransferSchedulerClient, WorkerSchedulerClient,
};

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, OnceLock};

use super::*;
use crate::block_manager::distributed::{get_leader_zmq_ack_url, get_leader_zmq_pub_url};
use crate::block_manager::vllm::connector::worker::event_sync_blocking;
use crate::{block_manager::distributed::VllmTensor, to_pyerr};
use dynamo_runtime::DistributedRuntime;

use crate::{
    extract_distributed_runtime_from_obj, get_current_cancel_token, get_current_tokio_handle,
};
use anyhow;
use dynamo_llm::block_manager::distributed::{KvbmWorker, KvbmWorkerConfig};
use dynamo_llm::block_manager::layout::LayoutType;
use dynamo_llm::block_manager::storage::torch::TorchTensor;
use dynamo_runtime::utils::task::CriticalTaskExecutionHandle;

pub trait Worker: Send + Sync {
    fn register_kv_caches(
        &mut self,
        num_device_blocks: usize,
        page_size: usize,
        device_id: usize,
        dtype_width_bytes: usize,
        kv_cache_tensor: Arc<VllmTensor>,
        raw_event_handles: Vec<u64>,
    ) -> anyhow::Result<()>;

    fn bind_connector_meta(&mut self, metadata: Vec<u8>) -> anyhow::Result<()>;

    fn start_load_kv(&mut self) -> anyhow::Result<()>;

    fn execute_offload_operations(&mut self) -> anyhow::Result<()>;

    fn save_kv_layer(&mut self, layer_idx: usize) -> anyhow::Result<()>;

    fn get_finished(
        &mut self,
        finished_gen_req_ids: Vec<u64>,
        started_loading_req_ids: Vec<u64>,
    ) -> (Vec<u64>, Vec<u64>);

    /// Submit offload operations to execute after the CUDA event completes (non-blocking).
    /// Does slot bookkeeping synchronously, then spawns an async task to poll the event
    /// and send operations to the scheduler when complete.
    fn submit_offload_on_event(&mut self, event: u64) -> anyhow::Result<()>;
}

pub struct KvConnectorWorker {
    _drt: Option<Arc<DistributedRuntime>>,
    kvbm_worker: OnceLock<KvbmWorker>,
    connector: WorkerSchedulerClient,
    transfer_client: TransferSchedulerClient,

    loads_done: HashSet<String>,
    stores_done: HashSet<String>,
    failed_requests: HashSet<String>,
    active_keys: HashMap<String, SlotKey>,
    local_epochs: HashMap<SlotKey, LocalEpochState>,

    onboarding_operations: Vec<WorkerTransferRequest>,
    offloading_operations: Vec<WorkerTransferRequest>,

    bound: bool,
    iteration: u64,
    layers_complete: usize,

    /// cuda events created by the python side
    layer_events: Vec<u64>,
}

#[derive(Debug, Clone, Default)]
struct LocalEpochState {
    onboarding_pending: bool,
    offloading_pending: bool,
}

impl KvConnectorWorker {
    fn new(drt: Option<Arc<DistributedRuntime>>, trtllm_rank: String) -> anyhow::Result<Self> {
        let runtime = get_current_tokio_handle();

        let (scheduler, worker_client, transfer_client) =
            Scheduler::new(trtllm_rank.clone(), get_current_cancel_token());

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
            "KvConnectorWorker initialized with worker_rank: {}",
            trtllm_rank
        );

        Ok(Self {
            _drt: drt,
            kvbm_worker: OnceLock::new(),
            connector: worker_client,
            transfer_client,
            loads_done: HashSet::new(),
            stores_done: HashSet::new(),
            failed_requests: HashSet::new(),
            active_keys: HashMap::new(),
            local_epochs: HashMap::new(),
            onboarding_operations: Vec::new(),
            offloading_operations: Vec::new(),
            bound: false,
            iteration: 0,
            layers_complete: 0,
            layer_events: Vec::new(),
        })
    }
}

impl Worker for KvConnectorWorker {
    fn register_kv_caches(
        &mut self,
        num_device_blocks: usize,
        page_size: usize,
        device_id: usize,
        dtype_width_bytes: usize,
        kv_cache_tensor: Arc<VllmTensor>,
        raw_event_handles: Vec<u64>,
    ) -> anyhow::Result<()> {
        if self.kvbm_worker.get().is_some() {
            tracing::warn!("kvbm worker already registered");
            return Err(anyhow::anyhow!("kvbm worker already registered"));
        }

        let kv_cache_tensors = vec![kv_cache_tensor as Arc<dyn TorchTensor>];

        let config = KvbmWorkerConfig::builder()
            .cancel_token(get_current_cancel_token())
            .num_device_blocks(num_device_blocks)
            .page_size(page_size)
            .tensors(kv_cache_tensors)
            .device_id(device_id)
            .dtype_width_bytes(dtype_width_bytes)
            .device_layout_type(LayoutType::FullyContiguous)
            .host_layout_type(LayoutType::FullyContiguous)
            .disk_layout_type(LayoutType::FullyContiguous)
            .leader_pub_url(get_leader_zmq_pub_url())
            .leader_ack_url(get_leader_zmq_ack_url())
            .scheduler_client(Some(self.transfer_client.clone()))
            .build()?;

        self.layer_events = raw_event_handles;

        let worker = get_current_tokio_handle().block_on(async move {
            let worker = KvbmWorker::new(config, true).await?;
            anyhow::Ok(worker)
        })?;

        self.kvbm_worker
            .set(worker)
            .map_err(|_| anyhow::anyhow!("failed to set kvbm worker"))?;

        Ok(())
    }

    fn bind_connector_meta(&mut self, metadata: Vec<u8>) -> anyhow::Result<()> {
        let metadata: ConnectorMetadata = serde_json::from_slice(&metadata)?;
        self.loads_done = metadata.loads_done.clone();
        self.stores_done = metadata.stores_done.clone();
        self.failed_requests = metadata.failed.clone();
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

        // local actions
        // - create a request slot for each new request
        // - for each action in the metadata, add the action to the request slot
        // - send the list of actions to the engine to track completion

        for slot_info in metadata.new_slots {
            self.active_keys
                .insert(slot_info.key.request_id.clone(), slot_info.key.clone());
            self.local_epochs.entry(slot_info.key.clone()).or_default();
            // Create slot with expected immediate ops count BEFORE any operations arrive.
            // This ensures proper completion tracking and avoids race conditions in TP>1.
            self.connector.create_slot_with_key_and_immediate_ops(
                slot_info.key,
                slot_info.expected_immediate_ops,
            )?;
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

        debug_assert!(
            self.onboarding_operations.is_empty(),
            "onboarding operations should be empty"
        );
        self.onboarding_operations = onboarding_operations;

        debug_assert!(
            self.offloading_operations.is_empty(),
            "offloading operations should be empty"
        );
        self.offloading_operations = offloading_operations;

        Ok(())
    }

    // Assumes the operations are in a valid state for offloading.
    fn execute_offload_operations(&mut self) -> anyhow::Result<()> {
        let offloading_operations = std::mem::take(&mut self.offloading_operations);
        for operation in offloading_operations {
            self.connector.enqueue_request(operation)?;
        }
        Ok(())
    }

    fn save_kv_layer(&mut self, _layer_idx: usize) -> anyhow::Result<()> {
        self.layers_complete += 1;
        if self.layers_complete == self.layer_events.len() {
            // block on the the completion of the last layer
            // todo(ryan): capture the context, pass this to the scheduler to do the await on another thread
            // or put the event on a stream and use stream waits to keep it all on device.
            event_sync_blocking(self.layer_events[self.layers_complete - 1]);
            if let Err(e) = self.execute_offload_operations() {
                tracing::error!("Failed to execute offload operations: {}", e);
            }
        }
        Ok(())
    }

    fn start_load_kv(&mut self) -> anyhow::Result<()> {
        let onboarding_operations = self.onboarding_operations.clone();
        for operation in onboarding_operations {
            let key = operation.key.clone();
            self.connector.enqueue_request(operation)?;
            self.local_epochs.entry(key).or_default().onboarding_pending = true;
        }
        Ok(())
    }

    fn get_finished(
        &mut self,
        finished_gen_req_ids: Vec<u64>,
        started_loading_req_ids: Vec<u64>,
    ) -> (Vec<u64>, Vec<u64>) {
        let mut is_finished_offloading = HashSet::new();
        let mut is_finished_onboarding = HashSet::new();

        for request_id in finished_gen_req_ids {
            tracing::debug!(request_id, "marking request as finished");

            let request_id = request_id.to_string();
            let Some(key) = self.active_keys.get(&request_id).cloned() else {
                tracing::warn!(
                    request_id = request_id,
                    "finished request received for unknown request_id; assuming never started"
                );
                continue;
            };
            self.local_epochs.entry(key).or_default().offloading_pending = true;
        }

        for request_id in started_loading_req_ids {
            let request_id = request_id.to_string();
            if let Some(key) = self.active_keys.get(&request_id).cloned() {
                self.local_epochs.entry(key).or_default().onboarding_pending = true;
            }
        }

        let offloading_keys: Vec<SlotKey> = self
            .local_epochs
            .iter()
            .filter_map(|(key, state)| state.offloading_pending.then_some(key.clone()))
            .collect();
        for key in offloading_keys {
            let stores_done = self.stores_done.contains(&key.request_id);
            if !self.connector.has_key(&key) || stores_done {
                is_finished_offloading.insert(key.request_id.clone());
            }
        }

        for request_id in &is_finished_offloading {
            if let Some(key) = self.active_keys.remove(request_id) {
                if self.connector.has_key(&key) {
                    if let Err(e) = self.connector.remove_key(&key) {
                        tracing::error!(
                            request_id,
                            generation = key.generation,
                            "failed to remove slot: {e}; scheduler disconnected"
                        );
                    }
                }
                self.local_epochs.remove(&key);
            }
        }

        let onboarding_keys: Vec<SlotKey> = self
            .local_epochs
            .iter()
            .filter_map(|(key, state)| state.onboarding_pending.then_some(key.clone()))
            .collect();
        for key in onboarding_keys {
            let loads_done = self.loads_done.contains(&key.request_id);
            if !self.connector.has_key(&key) || loads_done {
                is_finished_onboarding.insert(key.request_id.clone());
            }
        }

        for request_id in &is_finished_onboarding {
            if let Some(key) = self.active_keys.get(request_id).cloned() {
                if self.connector.has_key(&key) {
                    if let Err(e) = self.connector.remove_key(&key) {
                        tracing::error!(
                            request_id,
                            generation = key.generation,
                            "failed to remove slot: {e}; scheduler disconnected"
                        );
                    }
                }
                if let Some(state) = self.local_epochs.get_mut(&key) {
                    state.onboarding_pending = false;
                }
            }
        }

        let finished_offloading: Vec<u64> = is_finished_offloading
            .iter()
            .filter_map(|s| s.parse::<u64>().ok()) // parse String -> u64
            .collect();

        let finished_onboarding: Vec<u64> = is_finished_onboarding
            .iter()
            .filter_map(|s| s.parse::<u64>().ok()) // parse String -> u64
            .collect();

        (finished_offloading, finished_onboarding)
    }

    fn submit_offload_on_event(&mut self, event: u64) -> anyhow::Result<()> {
        let operations = std::mem::take(&mut self.offloading_operations);

        let tx = self.connector.get_scheduler_tx();

        // Use std::thread since we may be in a subprocess without tokio runtime
        std::thread::spawn(move || {
            // Block this thread until event completes (doesn't block main thread)
            event_sync_blocking(event);

            // Send operations to scheduler
            for op in operations {
                if let Err(e) = tx.send(SchedulerMessage::EnqueueRequest(op)) {
                    tracing::error!("Failed to send offload operation: {}", e);
                }
            }
        });

        Ok(())
    }
}

#[pyclass]
pub struct PyTrtllmKvConnectorWorker {
    connector_worker: Box<dyn Worker>,
}

#[pymethods]
impl PyTrtllmKvConnectorWorker {
    #[new]
    #[pyo3(signature = (py_drt, trtllm_rank))]
    pub fn new(py_drt: Option<PyObject>, trtllm_rank: String) -> PyResult<Self> {
        let drt: Option<Arc<DistributedRuntime>> = Python::with_gil(|py| {
            if let Some(obj) = py_drt {
                extract_distributed_runtime_from_obj(py, obj)
            } else {
                Ok(None)
            }
        })?;

        let connector_worker: Box<dyn Worker> =
            Box::new(KvConnectorWorker::new(drt, trtllm_rank).map_err(to_pyerr)?);
        Ok(Self { connector_worker })
    }

    pub fn register_kv_caches(
        &mut self,
        num_device_blocks: usize,
        page_size: usize,
        device_id: usize,
        dtype_width_bytes: usize,
        kv_cache_tensor: Py<PyAny>,
        raw_event_handles: Vec<u64>,
    ) -> PyResult<()> {
        // Convert Python tensor to Rust VllmTensor objects
        let rust_kv_cache_tensor = Arc::new(VllmTensor::new(kv_cache_tensor).map_err(to_pyerr)?);

        self.connector_worker
            .register_kv_caches(
                num_device_blocks,
                page_size,
                device_id,
                dtype_width_bytes,
                rust_kv_cache_tensor,
                raw_event_handles,
            )
            .map_err(to_pyerr)
    }

    pub fn bind_connector_meta(&mut self, metadata: Vec<u8>) -> PyResult<()> {
        self.connector_worker
            .bind_connector_meta(metadata)
            .map_err(to_pyerr)
    }

    pub fn execute_offload_operations(&mut self) -> PyResult<()> {
        self.connector_worker
            .execute_offload_operations()
            .map_err(to_pyerr)
    }

    pub fn save_kv_layer(&mut self, layer_idx: usize) -> PyResult<()> {
        self.connector_worker
            .save_kv_layer(layer_idx)
            .map_err(to_pyerr)
    }

    pub fn start_load_kv(&mut self) -> PyResult<()> {
        self.connector_worker.start_load_kv().map_err(to_pyerr)
    }

    pub fn get_finished(
        &mut self,
        finished_gen_req_ids: Vec<u64>,
        started_loading_req_ids: Vec<u64>,
    ) -> (Vec<u64>, Vec<u64>) {
        self.connector_worker
            .get_finished(finished_gen_req_ids, started_loading_req_ids)
    }

    pub fn submit_offload_on_event(&mut self, event: u64) -> PyResult<()> {
        self.connector_worker
            .submit_offload_on_event(event)
            .map_err(to_pyerr)
    }
}
