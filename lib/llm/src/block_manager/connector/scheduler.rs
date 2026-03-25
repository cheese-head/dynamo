// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashSet;
use std::sync::atomic::{AtomicU64, Ordering};

use super::protocol::*;
use super::*;

use tokio::sync::mpsc;

pub const DISCONNECTED_WARNING: &str =
    "runtime error: connections between components were lost; likely tearing down";

#[derive(Debug, thiserror::Error)]
pub enum SchedulerError {
    #[error("runtime error: connections between components were lost; likely tearing down")]
    Disconnected,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SchedulingDecision {
    Execute,
    Cancel,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EpochPhase {
    Onboarding,
    Active,
    Draining,
    Closed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CloseReason {
    Completed,
    Cancelled,
    Failed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ImmediateResultOutcome {
    Applied,
    Duplicate,
    Buffered,
    Stale,
}

/// A client for the scheduler. One-time use. Capture a clone per task.
#[derive(Clone)]
pub struct TransferSchedulerClient {
    scheduler_tx: mpsc::Sender<TransferToSchedulerMessage>,
}

impl TransferSchedulerClient {
    pub fn new(scheduler_tx: mpsc::Sender<TransferToSchedulerMessage>) -> Self {
        Self { scheduler_tx }
    }

    /// If the [SchedulingDecision::Execute] is returned, the caller receives a completion handle.
    /// The completion handle be marked as completed after the
    ///
    /// If the [SchedulingDecision::Cancel] is returned, the transfer is cancelled and the completion handle
    /// must not be dropped.
    #[tracing::instrument(level = "debug", skip_all, fields(request_id = %request.key.request_id, generation = request.key.generation, operation_id = %request.uuid))]
    pub async fn schedule_transfer(
        self,
        request: LeaderTransferRequest,
    ) -> anyhow::Result<Box<dyn TransferCompletionHandle>> {
        let scheduler_tx = self.scheduler_tx.clone();
        match request.request_type {
            RequestType::Immediate => {
                let handle = ImmediateTransferCompletionHandle::new(
                    request.key,
                    request.uuid,
                    request.chained,
                    scheduler_tx.clone(),
                );
                Ok(Box::new(handle))
            }
            RequestType::Scheduled => {
                let (response_tx, response_rx) = oneshot::channel();
                let request = TransferScheduleRequest {
                    leader_request: request,
                    response_tx,
                };

                tracing::debug!("sending schedule request to scheduler");
                scheduler_tx
                    .send(TransferToSchedulerMessage::ScheduleRequest(request))
                    .await?;

                tracing::debug!("awaiting response from scheduler");
                let handle = response_rx.await?.wait_for_decision().await;

                tracing::debug!(
                    "received scheduler decision: {:?}",
                    handle.scheduler_decision()
                );
                Ok(handle)
            }
        }
    }
}

pub struct WorkerSchedulerClient {
    slots: HashMap<SlotKey, WorkerSchedulerClientSlot>,
    worker_id: String,
    scheduler_tx: mpsc::UnboundedSender<SchedulerMessage>,
    /// Receiver for failure notifications from the scheduler
    failure_rx: mpsc::UnboundedReceiver<(String, uuid::Uuid)>,
    iteration: u64,
    iteration_complete: bool,
    layers_complete: u32,
}

impl WorkerSchedulerClient {
    pub fn new(
        worker_id: String,
        scheduler_tx: mpsc::UnboundedSender<SchedulerMessage>,
        failure_rx: mpsc::UnboundedReceiver<(String, uuid::Uuid)>,
        _cancel_token: CancellationToken,
    ) -> Self {
        Self {
            slots: HashMap::new(),
            worker_id,
            scheduler_tx,
            failure_rx,
            iteration: 0,
            iteration_complete: true,
            layers_complete: 0,
        }
    }

    pub fn iteration(&self) -> u64 {
        self.iteration
    }

    pub fn start_next_iteration(&mut self) -> Result<(), SchedulerError> {
        // debug_assert!(
        //     self.iteration_complete,
        //     "previous iteration must be complete before starting a new iteration"
        // );
        self.iteration += 1;
        self.iteration_complete = false;
        self.layers_complete = 0;
        self.scheduler_tx
            .send(SchedulerMessage::StartIteration(self.iteration))
            .map_err(|_| SchedulerError::Disconnected)
    }

    pub fn mark_layer_complete(&mut self, layer_name: String) -> Result<(), SchedulerError> {
        debug_assert!(
            !self.iteration_complete,
            "iteration must be complete before marking a layer as complete"
        );
        self.layers_complete += 1;
        self.scheduler_tx
            .send(SchedulerMessage::UpdateLayersCompleted(
                layer_name,
                self.layers_complete,
            ))
            .map_err(|_| SchedulerError::Disconnected)
    }

    pub fn mark_iteration_complete(&mut self) -> Result<(), SchedulerError> {
        debug_assert!(
            !self.iteration_complete,
            "iteration must be complete before marking it as complete"
        );
        self.iteration_complete = true;
        self.scheduler_tx
            .send(SchedulerMessage::EndIteration(self.iteration))
            .map_err(|_| SchedulerError::Disconnected)
    }
}

#[derive(Debug, Default)]
pub struct WorkerSchedulerClientSlot {
    operations: Vec<uuid::Uuid>,
    completed: Arc<AtomicU64>,
}

impl WorkerSchedulerClientSlot {
    fn new() -> Self {
        Self {
            operations: Vec::new(),
            completed: Arc::new(AtomicU64::new(0)),
        }
    }

    fn make_scheduler_slot_request(
        &self,
        key: SlotKey,
        worker_id: String,
        expected_immediate_ops: u64,
    ) -> SchedulerCreateSlotDetails {
        SchedulerCreateSlotDetails {
            key,
            worker_id,
            completed: self.completed.clone(),
            expected_immediate_ops,
        }
    }

    pub fn is_complete(&self) -> bool {
        // Use Acquire to synchronize with Release in handle_immediate_result
        self.completed.load(Ordering::Acquire) == self.operations.len() as u64
    }
}

impl WorkerSchedulerClient {
    /// Create a slot with the expected number of immediate (onboard) operations.
    /// This count is used to properly track completion and must match the number of
    /// ImmediateTransferResult messages that will be received.
    pub fn create_slot_with_key_and_immediate_ops(
        &mut self,
        key: SlotKey,
        expected_immediate_ops: u64,
    ) -> Result<(), SchedulerError> {
        // create a request slot
        let slot = WorkerSchedulerClientSlot::new();
        let request = slot.make_scheduler_slot_request(
            key.clone(),
            self.worker_id.clone(),
            expected_immediate_ops,
        );

        // insert the slot into the local worker slots map
        self.slots.insert(key.clone(), slot);

        // send a request to insert the slot into the engine state
        self.scheduler_tx
            .send(SchedulerMessage::CreateSlot(request))
            .map_err(|_| SchedulerError::Disconnected)?;
        Ok(())
    }

    pub fn has_key(&self, key: &SlotKey) -> bool {
        self.slots.contains_key(key)
    }

    pub fn is_key_complete(&self, key: &SlotKey) -> bool {
        match self.slots.get(key) {
            Some(slot) => slot.is_complete(),
            None => true,
        }
    }

    pub fn remove_key(&mut self, key: &SlotKey) -> Result<(), SchedulerError> {
        let Some(slot) = self.slots.remove(key) else {
            tracing::warn!(request_id = %key, "remove_key: slot already removed, skipping");
            return Ok(());
        };
        assert!(slot.is_complete());
        self.scheduler_tx
            .send(SchedulerMessage::RequestFinished(
                SchedulerRemoveSlotDetails {
                    key: key.clone(),
                    worker_id: self.worker_id.clone(),
                },
            ))
            .map_err(|_| SchedulerError::Disconnected)
    }

    /// Enqueues a request to the scheduler.
    ///
    /// Both the worker client and the scheduler keep track of outstanding requests.
    /// The atomic counter to mark completion is shared, but only incremented by the scheduler.
    pub fn enqueue_request(
        &mut self,
        request: WorkerTransferRequest,
    ) -> Result<(), SchedulerError> {
        let slot = match self.slots.get_mut(&request.key) {
            Some(slot) => slot,
            None => {
                tracing::warn!(
                    request_id = %request.key,
                    "slot does not exist (may have been cleared while forward pass in-flight), skipping"
                );
                return Ok(());
            }
        };

        slot.operations.push(request.uuid);

        match request.request_type {
            RequestType::Immediate => {}
            RequestType::Scheduled => {
                self.scheduler_tx
                    .send(SchedulerMessage::EnqueueRequest(request))
                    .map_err(|_| SchedulerError::Disconnected)?;
            }
        }
        Ok(())
    }

    /// Clone the scheduler channel for async use.
    pub fn get_scheduler_tx(&self) -> mpsc::UnboundedSender<SchedulerMessage> {
        self.scheduler_tx.clone()
    }

    /// Record operation in slot (bookkeeping only, no send).
    /// This updates the slot's expected operation count so is_complete() works correctly.
    pub fn record_operation_key(&mut self, key: &SlotKey, uuid: uuid::Uuid) {
        match self.slots.get_mut(key) {
            Some(slot) => slot.operations.push(uuid),
            None => tracing::warn!(
                request_id = %key,
                "record_operation_key: slot does not exist (cleared mid-flight), skipping"
            ),
        }
    }

    /// Drain all pending failure notifications from the scheduler (non-blocking).
    /// Returns failures grouped by request_id.
    pub fn drain_failures(&mut self) -> HashMap<String, HashSet<uuid::Uuid>> {
        let mut failures: HashMap<String, HashSet<uuid::Uuid>> = HashMap::new();
        while let Ok((request_id, uuid)) = self.failure_rx.try_recv() {
            failures.entry(request_id).or_default().insert(uuid);
        }
        failures
    }
}

pub type Iteration = u64;
pub type LayerName = String;
pub type LayerIndex = u32;

pub enum SchedulerMessage {
    /// Issued by worker to create a shared request state between worker and scheduler
    CreateSlot(SchedulerCreateSlotDetails),

    /// Enqueue a worker requested operation to the scheduler, this is one-half of the necessary
    /// bits to enqueu the operation. The other half is leader driven and propagated to the scheduler
    /// via the [TransferScheduleRequest]
    EnqueueRequest(WorkerTransferRequest),

    /// Issued at the start of a forward pass iteration
    StartIteration(Iteration),

    /// Issued at the end of a forward pass iteration, with the iteration number
    EndIteration(Iteration),

    /// Issued by the leader to update the number of layers completed
    UpdateLayersCompleted(LayerName, LayerIndex),

    /// Worker received a notification that its binding to the given request epoch has completed.
    RequestFinished(SchedulerRemoveSlotDetails),
}

enum SchedulerEvent {
    StartIteration(Iteration),
    EndIteration(Iteration),
    UpdateLayersCompleted(LayerName, LayerIndex),
    CreateSlot(SchedulerCreateSlotDetails),
    RequestFinished(SchedulerRemoveSlotDetails),
    EnqueueRequest(WorkerTransferRequest),
    ScheduleRequest(TransferScheduleRequest),
    ImmediateResult(ImmediateTransferResult),
}

enum SchedulerEffect {
    ScheduleTransfer(ScheduledTaskController),
    IncrementBindings(SlotKey, u64),
    NotifyFailure(String, uuid::Uuid),
}

pub struct Scheduler {
    // Authoritative logical epochs keyed by SlotKey.
    epochs: HashMap<SlotKey, RequestEpoch>,

    // Latest observed generation per request_id.
    latest_generation: HashMap<String, u64>,

    // Tombstones for recently closed epochs so late events can be classified safely.
    tombstones: HashMap<SlotKey, ClosedEpochMeta>,

    // Created during the responses to a scheduled transfer request.
    // Note: this does not require a worker slot binding to exist yet.
    cancel_tokens: HashMap<SlotKey, CancellationToken>,

    // Buffered immediate results keyed by logical epoch.
    pending_immediate_results: HashMap<SlotKey, HashSet<uuid::Uuid>>,

    // This object coordinates the two-stage execution of a scheduled transfer request.
    // If the scheduled request arrives first, the controller object will be Some; otherwise,
    // the worker-side request arrived first and it will be None.
    enqueued_requests: HashMap<SlotKey, HashMap<uuid::Uuid, TransferRequestSource>>,

    // Messages from the worker arrive on this channel
    worker_rx: mpsc::UnboundedReceiver<SchedulerMessage>,

    // Messages from the transfer client arrive on this channel
    transfer_rx: mpsc::Receiver<TransferToSchedulerMessage>,

    /// Sender for failure notifications to the worker (non-blocking)
    failure_tx: mpsc::UnboundedSender<(String, uuid::Uuid)>,

    iteration: u64,
    layers_complete: u32,
    iteration_complete: bool,
}

impl Scheduler {
    fn update_epoch_phase(epoch: &mut RequestEpoch) {
        if epoch.close_reason.is_some() {
            epoch.phase = EpochPhase::Closed;
        } else if matches!(epoch.phase, EpochPhase::Draining) {
            return;
        } else if epoch.completed_immediate_ops < epoch.expected_immediate_ops {
            epoch.phase = EpochPhase::Onboarding;
        } else {
            epoch.phase = EpochPhase::Active;
        }
    }

    fn is_stale_key(&self, key: &SlotKey) -> bool {
        self.latest_generation
            .get(&key.request_id)
            .is_some_and(|latest| key.generation < *latest)
    }

    pub fn new(
        worker_id: String,
        cancel_token: CancellationToken,
    ) -> (Self, WorkerSchedulerClient, TransferSchedulerClient) {
        let (scheduler_tx, scheduler_rx) = mpsc::unbounded_channel();
        let (transfer_tx, transfer_rx) = mpsc::channel(128);
        let (failure_tx, failure_rx) = mpsc::unbounded_channel();
        let worker_client =
            WorkerSchedulerClient::new(worker_id, scheduler_tx, failure_rx, cancel_token);
        let transfer_client = TransferSchedulerClient::new(transfer_tx);
        (
            Scheduler {
                epochs: HashMap::new(),
                latest_generation: HashMap::new(),
                tombstones: HashMap::new(),
                cancel_tokens: HashMap::new(),
                pending_immediate_results: HashMap::new(),
                enqueued_requests: HashMap::new(),
                worker_rx: scheduler_rx,
                transfer_rx,
                failure_tx,
                iteration: 0,
                layers_complete: 0,
                iteration_complete: true,
            },
            worker_client,
            transfer_client,
        )
    }

    pub async fn run(&mut self) -> anyhow::Result<()> {
        loop {
            if !self.step().await {
                break;
            }
        }
        tracing::warn!(
            iteration = self.iteration,
            epochs = self.epochs.len(),
            "scheduler exiting: worker or transfer channel closed"
        );
        Ok(())
    }

    async fn step(&mut self) -> bool {
        if self.worker_rx.is_closed() || self.transfer_rx.is_closed() {
            return false;
        }

        tokio::select! {
            maybe_worker_msg = self.worker_rx.recv(), if !self.worker_rx.is_closed() => {
                match maybe_worker_msg {
                    Some(SchedulerMessage::StartIteration(new_iteration)) => {
                        let effects = self.apply(SchedulerEvent::StartIteration(new_iteration));
                        self.run_effects(effects);
                    }
                    Some(SchedulerMessage::EndIteration(iteration)) => {
                        let effects = self.apply(SchedulerEvent::EndIteration(iteration));
                        self.run_effects(effects);
                    }
                    Some(SchedulerMessage::UpdateLayersCompleted(last_layer_name, layers_completed)) => {
                        let effects = self.apply(SchedulerEvent::UpdateLayersCompleted(last_layer_name, layers_completed));
                        self.run_effects(effects);
                    }
                    Some(SchedulerMessage::CreateSlot(request)) => {
                        let effects = self.apply(SchedulerEvent::CreateSlot(request));
                        self.run_effects(effects);
                    }
                    Some(SchedulerMessage::RequestFinished(request)) => {
                        let effects = self.apply(SchedulerEvent::RequestFinished(request));
                        self.run_effects(effects);
                    }
                    Some(SchedulerMessage::EnqueueRequest(request)) => {
                        let effects = self.apply(SchedulerEvent::EnqueueRequest(request));
                        self.run_effects(effects);
                    }
                    None => {
                        return false;
                    }
                }
            }
            maybe_transfer_msg = self.transfer_rx.recv(), if !self.transfer_rx.is_closed() => {
                match maybe_transfer_msg {
                    Some(TransferToSchedulerMessage::ScheduleRequest(request)) => {
                        let effects = self.apply(SchedulerEvent::ScheduleRequest(request));
                        self.run_effects(effects);
                    }
                    Some(TransferToSchedulerMessage::ImmediateResult(result)) => {
                        let effects = self.apply(SchedulerEvent::ImmediateResult(result));
                        self.run_effects(effects);
                    }
                    None => {
                        return false;
                    }
                }
             }
        }
        true
    }

    fn apply(&mut self, event: SchedulerEvent) -> Vec<SchedulerEffect> {
        match event {
            SchedulerEvent::StartIteration(iteration) => {
                self.start_iteration(iteration);
                Vec::new()
            }
            SchedulerEvent::EndIteration(iteration) => {
                self.end_iteration(iteration);
                Vec::new()
            }
            SchedulerEvent::UpdateLayersCompleted(layer_name, layers_completed) => {
                self.update_layers_completed(layer_name, layers_completed);
                Vec::new()
            }
            SchedulerEvent::CreateSlot(request) => {
                let key = request.key.clone();
                let request_id = key.request_id.clone();

                self.latest_generation
                    .entry(request_id.clone())
                    .and_modify(|generation| {
                        if *generation < key.generation {
                            *generation = key.generation;
                        }
                    })
                    .or_insert(key.generation);

                let slot = SchedulerSlot {
                    worker_id: request.worker_id,
                    completed: request.completed,
                };

                let num_buffered = self
                    .pending_immediate_results
                    .get(&key)
                    .map(|buffered_results| buffered_results.len() as u64)
                    .unwrap_or(0);

                let epoch = self
                    .epochs
                    .entry(key.clone())
                    .or_insert_with(|| RequestEpoch {
                        key: key.clone(),
                        phase: EpochPhase::Onboarding,
                        close_reason: None,
                        expected_immediate_ops: request.expected_immediate_ops,
                        completed_immediate_ops: 0,
                        seen_immediate_ops: HashSet::new(),
                        bindings: HashMap::new(),
                    });
                epoch.expected_immediate_ops = request.expected_immediate_ops;
                epoch.completed_immediate_ops = epoch.completed_immediate_ops.max(num_buffered);

                slot.completed
                    .store(epoch.completed_immediate_ops, Ordering::Release);
                epoch.bindings.insert(slot.worker_id.clone(), slot);
                Self::update_epoch_phase(epoch);
                Vec::new()
            }
            SchedulerEvent::RequestFinished(request) => {
                let key = request.key;
                if self.is_stale_key(&key) {
                    return Vec::new();
                }

                let request_id = key.request_id.clone();
                let Some(epoch) = self.epochs.get_mut(&key) else {
                    debug_assert!(false, "slot not found");
                    return Vec::new();
                };

                epoch.bindings.remove(&request.worker_id);

                if !epoch.bindings.is_empty() {
                    epoch.phase = EpochPhase::Draining;
                    tracing::debug!(
                        request_id,
                        generation = key.generation,
                        worker_id = %request.worker_id,
                        remaining_bindings = epoch.bindings.len(),
                        "worker binding removed; epoch still draining"
                    );
                    return Vec::new();
                }

                epoch.close_reason = Some(CloseReason::Completed);
                epoch.phase = EpochPhase::Closed;

                self.cancel_tokens.remove(&key);
                self.epochs.remove(&key);
                self.tombstones.insert(
                    key.clone(),
                    ClosedEpochMeta {
                        phase: EpochPhase::Closed,
                        close_reason: CloseReason::Completed,
                    },
                );

                if let Some(pending) = self.enqueued_requests.remove(&key)
                    && !pending.is_empty()
                {
                    tracing::warn!(
                        request_id,
                        num_pending = pending.len(),
                        "removing slot with un-paired scheduled operations; cancelling"
                    );
                }

                self.pending_immediate_results.remove(&key);

                tracing::debug!(
                    request_id,
                    iteration = self.iteration,
                    "engine state removing slot"
                );
                Vec::new()
            }
            SchedulerEvent::EnqueueRequest(request) => {
                if let Some(epoch) = self.epochs.get_mut(&request.key) {
                    Self::update_epoch_phase(epoch);
                } else {
                    debug_assert!(false, "slot does not exist");
                }

                let maybe_controller = self.try_prepare_controller(
                    request.key,
                    request.uuid,
                    TransferRequestSource::Worker,
                );

                maybe_controller
                    .map(|controller| vec![SchedulerEffect::ScheduleTransfer(controller)])
                    .unwrap_or_default()
            }
            SchedulerEvent::ScheduleRequest(request) => {
                let controller = self.process_scheduled_transfer_request(request).unwrap();

                let maybe_controller = self.try_prepare_controller(
                    controller.request.key.clone(),
                    controller.request.uuid,
                    TransferRequestSource::Transfer(controller),
                );

                maybe_controller
                    .map(|controller| {
                        tracing::debug!("scheduling transfer");
                        vec![SchedulerEffect::ScheduleTransfer(controller)]
                    })
                    .unwrap_or_default()
            }
            SchedulerEvent::ImmediateResult(result) => {
                let mut effects = Vec::new();
                if result.status.is_err() {
                    tracing::warn!(
                        request_id = %result.key.request_id,
                        operation_id = %result.uuid,
                        error = ?result.status,
                        "Immediate transfer failed"
                    );
                    effects.push(SchedulerEffect::NotifyFailure(
                        result.key.request_id.clone(),
                        result.uuid,
                    ));
                }

                if result.chained {
                    tracing::debug!("chained operation completed; skipping counter increment");
                    return effects;
                }

                match self.record_immediate_result(result.key.clone(), result.uuid) {
                    ImmediateResultOutcome::Applied => {
                        effects.push(SchedulerEffect::IncrementBindings(result.key.clone(), 1));
                    }
                    ImmediateResultOutcome::Duplicate => {
                        tracing::debug!("duplicate immediate result; ignoring");
                    }
                    ImmediateResultOutcome::Buffered => {
                        tracing::debug!("no slot found; buffering immediate result by SlotKey");
                    }
                    ImmediateResultOutcome::Stale => {
                        tracing::debug!("stale immediate result; dropping");
                    }
                }
                effects
            }
        }
    }

    fn run_effects(&mut self, effects: Vec<SchedulerEffect>) {
        for effect in effects {
            match effect {
                SchedulerEffect::ScheduleTransfer(controller) => self.schedule_request(controller),
                SchedulerEffect::IncrementBindings(key, delta) => {
                    if let Some(epoch) = self.epochs.get(&key) {
                        for slot in epoch.bindings.values() {
                            slot.completed.fetch_add(delta, Ordering::Release);
                        }
                    }
                }
                SchedulerEffect::NotifyFailure(request_id, uuid) => {
                    let _ = self.failure_tx.send((request_id, uuid));
                }
            }
        }
    }

    fn start_iteration(&mut self, iteration: u64) {
        // tracing::debug!(iteration, "engine state updating iteration");
        // debug_assert!(
        //     self.iteration_complete,
        //     "previous iteration must be complete before starting a new iteration"
        // );
        debug_assert_eq!(
            self.iteration,
            iteration - 1,
            "iteration must be incremented by 1"
        );
        self.iteration = iteration;
        self.layers_complete = 0;
        self.iteration_complete = false;
    }

    fn end_iteration(&mut self, iteration: u64) {
        tracing::debug!(iteration, "engine state updating iteration");
        self.iteration_complete = true;
    }

    fn update_layers_completed(&mut self, last_layer_name: String, layers_completed: u32) {
        self.layers_complete = layers_completed;
        tracing::debug!(
            iteration = self.iteration,
            layers_completed,
            "layer {last_layer_name} is complete"
        );
    }

    fn record_immediate_result(
        &mut self,
        key: SlotKey,
        uuid: uuid::Uuid,
    ) -> ImmediateResultOutcome {
        if self.is_stale_key(&key) {
            return ImmediateResultOutcome::Stale;
        }

        if self.tombstones.contains_key(&key) {
            return ImmediateResultOutcome::Stale;
        }

        if let Some(epoch) = self.epochs.get_mut(&key) {
            if !epoch.seen_immediate_ops.insert(uuid) {
                return ImmediateResultOutcome::Duplicate;
            }
            epoch.completed_immediate_ops += 1;
            Self::update_epoch_phase(epoch);
            return ImmediateResultOutcome::Applied;
        }

        self.pending_immediate_results
            .entry(key)
            .or_default()
            .insert(uuid);
        ImmediateResultOutcome::Buffered
    }

    /// This function is used to handle the request from worker or transfer based on their arrival order.
    /// It returns Some(ScheduledTaskController) if both worker and transfer have arrived, or None if any of them has not arrived yet.
    ///
    /// More details:
    /// If no uuid is found in enqueued_requests, it means neither worker nor transfer has arrived yet.
    /// Then, we will insert controller into enqueued_requests (for transfer) or None (for worker) and return None.
    ///
    /// If uuid is found in enqueued_requests, it means either worker or transfer has arrived.
    /// Then, we check the incoming controller. If it is Some, it means worker has arrived first and we can return it.
    /// If it is None, it means the transfer has arrived first and we can return the existing controller.
    fn try_prepare_controller(
        &mut self,
        key: SlotKey,
        uuid: uuid::Uuid,
        incoming: TransferRequestSource,
    ) -> Option<ScheduledTaskController> {
        let entry = self.enqueued_requests.entry(key).or_default();
        match (entry.remove(&uuid), incoming) {
            (Some(TransferRequestSource::Worker), TransferRequestSource::Transfer(controller)) => {
                tracing::debug!("worker arrived first, then transfer ==> scheduling transfer");
                Some(controller)
            }
            (Some(TransferRequestSource::Transfer(controller)), TransferRequestSource::Worker) => {
                tracing::debug!("transfer arrived first, then worker ==> scheduling transfer");
                Some(controller)
            }
            (None, TransferRequestSource::Worker) => {
                tracing::debug!("worker arrived first; must wait for transfer");
                entry.insert(uuid, TransferRequestSource::Worker);
                None
            }
            (None, TransferRequestSource::Transfer(controller)) => {
                tracing::debug!("transfer arrived first; must wait for worker");
                entry.insert(uuid, TransferRequestSource::Transfer(controller));
                None
            }
            _ => {
                panic!("invalid combination of request sources");
            }
        }
    }

    // this function will be a scheduler and will dispatch requests to be executed
    fn schedule_request(&mut self, xfer_req: ScheduledTaskController) {
        // tokio spawn execute_scheduled_transfer for first impl.  add fanciness later.
        self.execute_scheduled_transfer(xfer_req);
    }

    // this function will execute a transfer request, monitor its completion, and increment its
    // atomic completion counter when finished.
    //
    // this must tokio spawn and an indpendent task
    fn execute_scheduled_transfer(&mut self, xfer_req: ScheduledTaskController) {
        let completed: Vec<Arc<AtomicU64>> = self
            .epochs
            .get(&xfer_req.request.key)
            .expect("slot not found")
            .bindings
            .values()
            .map(|slot| slot.completed.clone())
            .collect();
        tokio::spawn(async move {
            let result = xfer_req.execute(SchedulingDecision::Execute).await;
            if result.is_ok() {
                for counter in completed {
                    counter.fetch_add(1, Ordering::Relaxed);
                }
            }
        });
    }

    /// Translate the [`TransferScheduleRequest`] into a local [`ScheduledTaskController`]
    /// This function returns to the transfer client the [`ScheduledTaskHandle`]
    fn process_scheduled_transfer_request(
        &mut self,
        xfer_req: TransferScheduleRequest,
    ) -> anyhow::Result<ScheduledTaskController> {
        // Create the next stage communcication p2p channel between scheduler and client
        let (decision_tx, decision_rx) = oneshot::channel();

        // Get or create the cancel token for this request
        let cancel_token = self
            .cancel_tokens
            .entry(xfer_req.leader_request.key.clone())
            .or_default()
            .child_token();

        // Create the ScheduledTaskHandle to send to the client
        let task_handle = ScheduledTaskHandle {
            decision_rx,
            cancel_token,
        };

        // Send the ScheduledTaskHandle back to the client side
        xfer_req
            .response_tx
            .send(task_handle)
            .map_err(|_| anyhow::anyhow!("Failed to send scheduled task handle to xfer client"))?;

        // Create the ScheduledTaskController to locally trigger the exection of the scheduled transfer task
        let controller = ScheduledTaskController {
            request: xfer_req.leader_request,
            decision_tx,
        };

        Ok(controller)
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ScheduledTaskError {}

pub struct ScheduledTaskController {
    request: LeaderTransferRequest,
    decision_tx: oneshot::Sender<(SchedulingDecision, oneshot::Sender<anyhow::Result<()>>)>,
}

impl ScheduledTaskController {
    pub async fn execute(self, decision: SchedulingDecision) -> anyhow::Result<()> {
        let (completion_tx, completion_rx) = oneshot::channel();
        self.decision_tx
            .send((decision, completion_tx))
            .map_err(|_| anyhow::anyhow!(DISCONNECTED_WARNING))?;
        let _ = completion_rx
            .await
            .map_err(|_| anyhow::anyhow!(DISCONNECTED_WARNING))?;
        Ok(())
    }
}

enum TransferRequestSource {
    Worker,
    Transfer(ScheduledTaskController),
}

pub struct ScheduledTaskAsyncResult {
    completion_rx: oneshot::Receiver<anyhow::Result<()>>,
}

impl ScheduledTaskAsyncResult {
    pub async fn await_completion(self) -> anyhow::Result<()> {
        self.completion_rx.await.unwrap()
    }
}

pub struct SchedulerCreateSlotDetails {
    pub key: SlotKey,
    pub worker_id: String,
    pub completed: Arc<AtomicU64>,
    /// Expected number of immediate (onboard) operations for this slot.
    pub expected_immediate_ops: u64,
}

pub struct SchedulerRemoveSlotDetails {
    pub key: SlotKey,
    pub worker_id: String,
}

struct RequestEpoch {
    key: SlotKey,
    phase: EpochPhase,
    close_reason: Option<CloseReason>,
    expected_immediate_ops: u64,
    completed_immediate_ops: u64,
    seen_immediate_ops: HashSet<uuid::Uuid>,
    bindings: HashMap<String, SchedulerSlot>,
}

struct ClosedEpochMeta {
    phase: EpochPhase,
    close_reason: CloseReason,
}

#[derive(Clone)]
pub struct SchedulerSlot {
    #[allow(dead_code)]
    worker_id: String,
    completed: Arc<AtomicU64>,
}

pub trait TaskScheduler {
    fn start_iteration(&mut self, iteration: u64) -> Result<(), SchedulerError>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    #[tokio::test]
    async fn test_scheduler_lifecycle() {
        let cancel_token = CancellationToken::new();
        let (mut scheduler, mut worker_client, _transfer_client) =
            Scheduler::new("worker-0".to_string(), cancel_token);
        let key = SlotKey::new("test".to_string(), 0);

        // create a slot
        worker_client
            .create_slot_with_key_and_immediate_ops(key.clone(), 0)
            .unwrap();

        // enqueue a request
        assert!(!scheduler.epochs.contains_key(&key));
        scheduler.step().await;
        assert!(scheduler.epochs.contains_key(&key));

        // test iteration triggers
        worker_client.start_next_iteration().unwrap();
        scheduler.step().await;
        assert_eq!(scheduler.iteration, 1);

        // test iteration end triggers
        worker_client.mark_iteration_complete().unwrap();
        scheduler.step().await;
        assert_eq!(scheduler.iteration, 1);
        assert!(scheduler.iteration_complete);
    }

    #[tokio::test]
    async fn test_transfer_immediate_arrives_first() {
        dynamo_runtime::logging::init();

        let cancel_token = CancellationToken::new();
        let (mut scheduler, mut worker_client, transfer_client) =
            Scheduler::new("worker-0".to_string(), cancel_token);
        let key = SlotKey::new("test".to_string(), 0);

        let operation_id = uuid::Uuid::new_v4();

        // on the transfer engine, a request arrives with a request type of immediate
        let request = LeaderTransferRequest {
            key: SlotKey::new("test".to_string(), 0),
            uuid: operation_id,
            requirement: None,
            request_type: RequestType::Immediate,
            chained: false,
        };

        let handle = transfer_client
            .clone()
            .schedule_transfer(request)
            .await
            .unwrap();

        // the transfer engine will immediately return a completion handle
        assert_eq!(handle.scheduler_decision(), SchedulingDecision::Execute);

        // the completion handle will be marked as complete
        handle.mark_complete(Ok(())).await;

        assert_eq!(scheduler.pending_immediate_results.len(), 0);
        scheduler.step().await;
        assert_eq!(scheduler.pending_immediate_results.len(), 1);

        // the request is completed - create slot with expected_immediate_ops=1
        worker_client
            .create_slot_with_key_and_immediate_ops(key.clone(), 1)
            .unwrap();

        assert!(!scheduler.epochs.contains_key(&key));
        scheduler.step().await;
        assert!(scheduler.epochs.contains_key(&key));

        // Buffered results are not removed in add_slot() - cleanup happens in remove_slot()
        // when the request finishes. This ensures all workers in TP>1 can have the buffered
        // count applied. The buffered count has already been applied to the slot's completed counter.
        assert_eq!(scheduler.pending_immediate_results.len(), 1);

        // neither the worker nor the scheduler should have observed the completion yet
        // this is because the worker has not yet requested it
        assert_eq!(
            scheduler
                .epochs
                .get(&key)
                .unwrap()
                .bindings
                .get("worker-0")
                .unwrap()
                .completed
                .load(Ordering::Relaxed),
            1
        );
        assert_eq!(
            worker_client
                .slots
                .get(&key)
                .unwrap()
                .completed
                .load(Ordering::Relaxed),
            1
        );

        // the worker has not issued any operations yet
        assert_eq!(worker_client.slots.get(&key).unwrap().operations.len(), 0);

        // enqueue the operation so is_complete() will return true (completed=1, operations.len()=1)
        let worker_request = WorkerTransferRequest {
            key: SlotKey::new("test".to_string(), 0),
            uuid: operation_id,
            transfer_type: TransferType::Load,
            request_type: RequestType::Immediate,
            block_ids: vec![],
        };
        worker_client.enqueue_request(worker_request).unwrap();
        assert_eq!(worker_client.slots.get(&key).unwrap().operations.len(), 1);
        assert!(worker_client.is_key_complete(&key));

        // verify that remove_slot() cleans up the buffered results
        assert_eq!(scheduler.pending_immediate_results.len(), 1);
        worker_client.remove_key(&key).unwrap();
        scheduler.step().await;

        // after remove_slot(), the buffered results should be cleaned up
        assert_eq!(scheduler.pending_immediate_results.len(), 0);
        assert!(!scheduler.epochs.contains_key(&key));
    }

    /// This test verifies that the scheduler can handle the case where the transfer engine's
    /// immediate result arrives after the worker has scheduled the operation.
    #[tokio::test]
    async fn test_transfer_immediate_arrives_last() {
        dynamo_runtime::logging::init();

        let cancel_token = CancellationToken::new();
        let (mut scheduler, mut worker_client, transfer_client) =
            Scheduler::new("worker-0".to_string(), cancel_token);
        let key = SlotKey::new("test".to_string(), 0);

        let operation_id = uuid::Uuid::new_v4();

        // on the transfer engine, a request arrives with a request type of immediate
        let request = LeaderTransferRequest {
            key: SlotKey::new("test".to_string(), 0),
            uuid: operation_id,
            requirement: None,
            request_type: RequestType::Immediate,
            chained: false,
        };

        let handle = transfer_client
            .clone()
            .schedule_transfer(request)
            .await
            .unwrap();

        // the transfer engine will immediately return a completion handle
        assert_eq!(handle.scheduler_decision(), SchedulingDecision::Execute);

        // assume this is a long running operation so our worker can enqueue the operation worker-side before the transfer-side completes
        worker_client
            .create_slot_with_key_and_immediate_ops(key.clone(), 0)
            .unwrap();
        assert!(!scheduler.epochs.contains_key(&key));
        scheduler.step().await;
        assert!(scheduler.epochs.contains_key(&key));
        assert_eq!(scheduler.pending_immediate_results.len(), 0);

        // the worker enqueues the operation
        let request = WorkerTransferRequest {
            key: SlotKey::new("test".to_string(), 0),
            uuid: operation_id,
            transfer_type: TransferType::Load,
            request_type: RequestType::Immediate,
            block_ids: vec![],
        };

        // immediate requests are not passed to the scheduler, but the completion will be automatically
        // visible on the client via the shared atomic counter
        worker_client.enqueue_request(request).unwrap();

        let worker_slot = worker_client.slots.get(&key).unwrap();
        assert_eq!(worker_slot.operations.len(), 1);
        assert_eq!(worker_slot.completed.load(Ordering::Relaxed), 0);

        // the completion handle will be marked as complete
        handle.mark_complete(Ok(())).await;

        assert_eq!(scheduler.pending_immediate_results.len(), 0);
        scheduler.step().await;
        assert_eq!(scheduler.pending_immediate_results.len(), 0);

        // neither the worker nor the scheduler should have observed the completion yet
        // this is because the worker has not yet requested it
        assert_eq!(
            scheduler
                .epochs
                .get(&key)
                .unwrap()
                .bindings
                .get("worker-0")
                .unwrap()
                .completed
                .load(Ordering::Relaxed),
            1
        );
        assert_eq!(
            worker_client
                .slots
                .get(&key)
                .unwrap()
                .completed
                .load(Ordering::Relaxed),
            1
        );

        // the worker has not issued any operations yet
        assert_eq!(worker_client.slots.get(&key).unwrap().operations.len(), 1);
    }

    // this test verifies that the scheduler can handle the case where the transfer engine's   /// in this case, the request arrives first via the worker client, meaning it traverse
    #[tokio::test]
    async fn test_transfer_scheduled_arrives_first() {
        dynamo_runtime::logging::init();

        let cancel_token = CancellationToken::new();
        let (mut scheduler, mut worker_client, transfer_client) =
            Scheduler::new("worker-0".to_string(), cancel_token);
        let key = SlotKey::new("test".to_string(), 0);

        let operation_id = uuid::Uuid::new_v4();

        // on the transfer engine, a request arrives with a request type of scheduled
        let request = LeaderTransferRequest {
            key: SlotKey::new("test".to_string(), 0),
            uuid: operation_id,
            requirement: None,
            request_type: RequestType::Scheduled,
            chained: false,
        };

        // transfer arrives first
        let handle = tokio::spawn(transfer_client.schedule_transfer(request));
        scheduler.step().await;

        // enqueued_requests should contain <request id, <uuid, and Some(controller)>> since transfer arrived first
        assert_eq!(
            scheduler
                .enqueued_requests
                .get(&SlotKey::new("test".to_string(), 0))
                .unwrap()
                .len(),
            1
        );
        assert!(matches!(
            scheduler
                .enqueued_requests
                .get(&SlotKey::new("test".to_string(), 0))
                .unwrap()
                .get(&operation_id),
            Some(TransferRequestSource::Transfer(_))
        ));

        worker_client
            .create_slot_with_key_and_immediate_ops(key.clone(), 0)
            .unwrap();
        assert!(!scheduler.epochs.contains_key(&key));
        scheduler.step().await;
        assert!(scheduler.epochs.contains_key(&key));

        let request = WorkerTransferRequest {
            key: SlotKey::new("test".to_string(), 0),
            uuid: operation_id,
            transfer_type: TransferType::Store,
            request_type: RequestType::Scheduled,
            block_ids: vec![],
        };

        // worker arrives last
        worker_client.enqueue_request(request).unwrap();
        scheduler.step().await;

        let handle = handle.await.unwrap().unwrap();
        handle.mark_complete(Ok(())).await;

        // after worker arrives, <uuid, and Some(controller)> inserted by transfer should be removed from enqueued_requests
        assert_eq!(
            scheduler
                .enqueued_requests
                .get(&SlotKey::new("test".to_string(), 0))
                .unwrap()
                .len(),
            0
        );

        // wait a bit to make sure the scheduled transfer to complete
        tokio::time::sleep(std::time::Duration::from_secs(1)).await;
        assert_eq!(
            worker_client
                .slots
                .get(&key)
                .unwrap()
                .completed
                .load(Ordering::Relaxed),
            1
        );
        assert_eq!(
            scheduler
                .epochs
                .get(&key)
                .unwrap()
                .bindings
                .get("worker-0")
                .unwrap()
                .completed
                .load(Ordering::Relaxed),
            1
        );

        // make sure all operations are complete
        assert!(worker_client.slots.get(&key).unwrap().is_complete());
    }

    #[tokio::test]
    async fn test_transfer_scheduled_arrives_last() {
        dynamo_runtime::logging::init();

        let cancel_token = CancellationToken::new();
        let (mut scheduler, mut worker_client, transfer_client) =
            Scheduler::new("worker-0".to_string(), cancel_token);
        let key = SlotKey::new("test".to_string(), 0);

        let operation_id = uuid::Uuid::new_v4();

        worker_client
            .create_slot_with_key_and_immediate_ops(key.clone(), 0)
            .unwrap();
        assert!(!scheduler.epochs.contains_key(&key));
        scheduler.step().await;
        assert!(scheduler.epochs.contains_key(&key));

        let request = WorkerTransferRequest {
            key: SlotKey::new("test".to_string(), 0),
            uuid: operation_id,
            transfer_type: TransferType::Store,
            request_type: RequestType::Scheduled,
            block_ids: vec![],
        };

        // worker arrives first
        worker_client.enqueue_request(request).unwrap();
        scheduler.step().await;

        // enqueued_requests should contain <request id, <uuid, and None>> since worker arrived first
        assert_eq!(
            scheduler
                .enqueued_requests
                .get(&SlotKey::new("test".to_string(), 0))
                .unwrap()
                .len(),
            1
        );
        assert!(matches!(
            scheduler
                .enqueued_requests
                .get(&SlotKey::new("test".to_string(), 0))
                .unwrap()
                .get(&operation_id),
            Some(TransferRequestSource::Worker)
        ));

        let request = LeaderTransferRequest {
            key: SlotKey::new("test".to_string(), 0),
            uuid: operation_id,
            requirement: None,
            request_type: RequestType::Scheduled,
            chained: false,
        };

        // transfer arrives last
        let handle = tokio::spawn(transfer_client.schedule_transfer(request));
        scheduler.step().await;
        let handle = handle.await.unwrap().unwrap();
        assert_eq!(handle.scheduler_decision(), SchedulingDecision::Execute);
        handle.mark_complete(Ok(())).await;

        // after transfer arrives, <uuid, and None> inserted by worker should be removed from enqueued_requests
        assert_eq!(
            scheduler
                .enqueued_requests
                .get(&SlotKey::new("test".to_string(), 0))
                .unwrap()
                .len(),
            0
        );

        // wait a bit to make sure the scheduled transfer to complete
        tokio::time::sleep(std::time::Duration::from_secs(1)).await;
        assert_eq!(
            worker_client
                .slots
                .get(&key)
                .unwrap()
                .completed
                .load(Ordering::Relaxed),
            1
        );
        assert_eq!(
            scheduler
                .epochs
                .get(&key)
                .unwrap()
                .bindings
                .get("worker-0")
                .unwrap()
                .completed
                .load(Ordering::Relaxed),
            1
        );

        // make sure all operations are complete
        assert!(worker_client.slots.get(&key).unwrap().is_complete());
    }

    #[tokio::test]
    async fn test_coordinate_scheduled_transfer_execution() {
        dynamo_runtime::logging::init();

        let cancel_token = CancellationToken::new();
        let (mut scheduler, _worker_client, transfer_client) =
            Scheduler::new("worker-0".to_string(), cancel_token);

        let operation_id = uuid::Uuid::new_v4();

        // Create a scheduled transfer request
        let request = LeaderTransferRequest {
            key: SlotKey::new("test".to_string(), 0),
            uuid: operation_id,
            requirement: None,
            request_type: RequestType::Scheduled,
            chained: false,
        };

        // allows us to pause the transfer task after the scheduler decision is made
        // but before the transfer is marked as complete
        let (got_handle_tx, got_handle_rx) = oneshot::channel();

        // Spawn the schedule_transfer call which will await our coordination function
        let _transfer_task = tokio::spawn(async move {
            let handle = transfer_client
                .clone()
                .schedule_transfer(request)
                .await
                .unwrap();

            got_handle_tx
                .send(handle)
                .map_err(|_| {
                    anyhow::anyhow!("failed to send handle back on testing oneshot channel")
                })
                .unwrap();
        });

        assert!(got_handle_rx.is_empty());

        // Simulate the scheduler making a decision and coordinating the execution
        // We skip that logic and go straight to the point we have a controller
        let controller = match scheduler.transfer_rx.recv().await {
            Some(msg) => match msg {
                TransferToSchedulerMessage::ScheduleRequest(schedule_req) => scheduler
                    .process_scheduled_transfer_request(schedule_req)
                    .ok(),
                _ => {
                    unreachable!("unexpected message type");
                }
            },
            None => {
                unreachable!("channel closed");
            }
        };

        // we still do not have both sides
        // we have the scheduler side controller, but we must trigger the controller to get a handle on the transfer engine
        let scheduler_controller = controller.expect("Expected a controller from the scheduler");
        assert!(got_handle_rx.is_empty());

        // Simulate some work being done - wait until the test releases us
        let completed = Arc::new(AtomicU64::new(0));
        let scheduler_result =
            tokio::spawn(scheduler_controller.execute(SchedulingDecision::Execute));

        // simulate the transfer engine receiving the decision
        let transfer_handle = got_handle_rx.await.unwrap();

        assert_eq!(
            transfer_handle.scheduler_decision(),
            SchedulingDecision::Execute
        );

        // Mark the transfer as complete with success
        transfer_handle.mark_complete(Ok(())).await;

        // wait for the scheduler to complete
        scheduler_result.await.unwrap().unwrap();
        // after the scheduler completes, the completed counter should be 1
        assert_eq!(completed.load(Ordering::Relaxed), 0);
    }

    #[tokio::test]
    async fn test_tp_bindings_are_keyed_by_worker_id() {
        let cancel_token = CancellationToken::new();
        let (mut scheduler, _worker_client, _transfer_client) =
            Scheduler::new("worker-0".to_string(), cancel_token);

        let key = SlotKey::new("test".to_string(), 0);
        let worker_0_completed = Arc::new(AtomicU64::new(0));
        let worker_1_completed = Arc::new(AtomicU64::new(0));

        let effects = scheduler.apply(SchedulerEvent::CreateSlot(SchedulerCreateSlotDetails {
            key: key.clone(),
            worker_id: "worker-0".to_string(),
            completed: worker_0_completed.clone(),
            expected_immediate_ops: 1,
        }));
        scheduler.run_effects(effects);
        let effects = scheduler.apply(SchedulerEvent::CreateSlot(SchedulerCreateSlotDetails {
            key: key.clone(),
            worker_id: "worker-1".to_string(),
            completed: worker_1_completed.clone(),
            expected_immediate_ops: 1,
        }));
        scheduler.run_effects(effects);

        let epoch = scheduler.epochs.get(&key).unwrap();
        assert_eq!(epoch.bindings.len(), 2);
        assert!(epoch.bindings.contains_key("worker-0"));
        assert!(epoch.bindings.contains_key("worker-1"));
        assert_eq!(worker_0_completed.load(Ordering::Relaxed), 0);
        assert_eq!(worker_1_completed.load(Ordering::Relaxed), 0);
    }

    #[tokio::test]
    async fn test_stale_immediate_result_is_dropped_after_new_generation() {
        let cancel_token = CancellationToken::new();
        let (mut scheduler, _worker_client, _transfer_client) =
            Scheduler::new("worker-0".to_string(), cancel_token);

        let old_key = SlotKey::new("test".to_string(), 0);
        let new_key = SlotKey::new("test".to_string(), 1);
        let completed = Arc::new(AtomicU64::new(0));

        let effects = scheduler.apply(SchedulerEvent::CreateSlot(SchedulerCreateSlotDetails {
            key: new_key.clone(),
            worker_id: "worker-0".to_string(),
            completed: completed.clone(),
            expected_immediate_ops: 1,
        }));
        scheduler.run_effects(effects);

        let effects = scheduler.apply(SchedulerEvent::ImmediateResult(ImmediateTransferResult {
            key: old_key.clone(),
            uuid: uuid::Uuid::new_v4(),
            status: Ok(()),
            chained: false,
        }));
        scheduler.run_effects(effects);

        assert!(!scheduler.pending_immediate_results.contains_key(&old_key));
        assert_eq!(completed.load(Ordering::Relaxed), 0);
        assert_eq!(
            scheduler
                .epochs
                .get(&new_key)
                .unwrap()
                .completed_immediate_ops,
            0
        );
    }

    #[tokio::test]
    async fn test_immediate_result_fans_out_to_all_epoch_bindings() {
        let cancel_token = CancellationToken::new();
        let (mut scheduler, _worker_client, _transfer_client) =
            Scheduler::new("worker-0".to_string(), cancel_token);

        let key = SlotKey::new("test".to_string(), 0);
        let worker_0_completed = Arc::new(AtomicU64::new(0));
        let worker_1_completed = Arc::new(AtomicU64::new(0));

        let effects = scheduler.apply(SchedulerEvent::CreateSlot(SchedulerCreateSlotDetails {
            key: key.clone(),
            worker_id: "worker-0".to_string(),
            completed: worker_0_completed.clone(),
            expected_immediate_ops: 1,
        }));
        scheduler.run_effects(effects);
        let effects = scheduler.apply(SchedulerEvent::CreateSlot(SchedulerCreateSlotDetails {
            key: key.clone(),
            worker_id: "worker-1".to_string(),
            completed: worker_1_completed.clone(),
            expected_immediate_ops: 1,
        }));
        scheduler.run_effects(effects);

        let effects = scheduler.apply(SchedulerEvent::ImmediateResult(ImmediateTransferResult {
            key: key.clone(),
            uuid: uuid::Uuid::new_v4(),
            status: Ok(()),
            chained: false,
        }));
        scheduler.run_effects(effects);

        assert_eq!(worker_0_completed.load(Ordering::Relaxed), 1);
        assert_eq!(worker_1_completed.load(Ordering::Relaxed), 1);
        assert_eq!(
            scheduler.epochs.get(&key).unwrap().phase,
            EpochPhase::Active
        );
    }

    #[tokio::test]
    async fn test_request_finished_detaches_one_binding_before_closing_epoch() {
        let cancel_token = CancellationToken::new();
        let (mut scheduler, _worker_client, _transfer_client) =
            Scheduler::new("worker-0".to_string(), cancel_token);

        let key = SlotKey::new("test".to_string(), 0);

        let effects = scheduler.apply(SchedulerEvent::CreateSlot(SchedulerCreateSlotDetails {
            key: key.clone(),
            worker_id: "worker-0".to_string(),
            completed: Arc::new(AtomicU64::new(1)),
            expected_immediate_ops: 0,
        }));
        scheduler.run_effects(effects);
        let effects = scheduler.apply(SchedulerEvent::CreateSlot(SchedulerCreateSlotDetails {
            key: key.clone(),
            worker_id: "worker-1".to_string(),
            completed: Arc::new(AtomicU64::new(1)),
            expected_immediate_ops: 0,
        }));
        scheduler.run_effects(effects);

        let effects = scheduler.apply(SchedulerEvent::RequestFinished(
            SchedulerRemoveSlotDetails {
                key: key.clone(),
                worker_id: "worker-0".to_string(),
            },
        ));
        scheduler.run_effects(effects);

        let epoch = scheduler.epochs.get(&key).unwrap();
        assert_eq!(epoch.phase, EpochPhase::Draining);
        assert_eq!(epoch.bindings.len(), 1);
        assert!(epoch.bindings.contains_key("worker-1"));

        let effects = scheduler.apply(SchedulerEvent::RequestFinished(
            SchedulerRemoveSlotDetails {
                key: key.clone(),
                worker_id: "worker-1".to_string(),
            },
        ));
        scheduler.run_effects(effects);

        assert!(!scheduler.epochs.contains_key(&key));
        assert!(scheduler.tombstones.contains_key(&key));
    }

    #[rstest]
    #[case(8, 2, 4)]
    #[case(32, 4, 8)]
    #[tokio::test(flavor = "multi_thread", worker_threads = 8)]
    async fn test_high_concurrency_epoch_fanout_and_stale_drop(
        #[case] num_requests: usize,
        #[case] num_bindings: usize,
        #[case] num_immediate_ops: usize,
    ) {
        let expected_messages =
            (num_requests * num_bindings) + (num_requests * num_immediate_ops * 2);

        let cancel_token = CancellationToken::new();
        let (mut scheduler, worker_client, transfer_client) =
            Scheduler::new("worker-0".to_string(), cancel_token);

        let scheduler_tx = worker_client.get_scheduler_tx();

        let mut completions: Vec<(SlotKey, Vec<Arc<AtomicU64>>)> = Vec::new();
        let mut join_handles = Vec::new();

        for request_index in 0..num_requests {
            let request_id = format!("req-{request_index}");
            let key = SlotKey::new(request_id.clone(), 1);
            let stale_key = SlotKey::new(request_id, 0);

            let binding_counters: Vec<_> = (0..num_bindings)
                .map(|_| Arc::new(AtomicU64::new(0)))
                .collect();
            completions.push((key.clone(), binding_counters.clone()));

            for (binding_index, counter) in binding_counters.into_iter().enumerate() {
                let tx = scheduler_tx.clone();
                let key = key.clone();
                join_handles.push(tokio::spawn(async move {
                    tx.send(SchedulerMessage::CreateSlot(SchedulerCreateSlotDetails {
                        key,
                        worker_id: format!("worker-{binding_index}"),
                        completed: counter,
                        expected_immediate_ops: num_immediate_ops as u64,
                    }))
                    .expect("failed to send create slot");
                }));
            }

            for _ in 0..num_immediate_ops {
                let current_client = transfer_client.clone();
                let current_key = key.clone();
                join_handles.push(tokio::spawn(async move {
                    let request = LeaderTransferRequest {
                        key: current_key,
                        uuid: uuid::Uuid::new_v4(),
                        requirement: None,
                        request_type: RequestType::Immediate,
                        chained: false,
                    };
                    let handle = current_client.schedule_transfer(request).await.unwrap();
                    handle.mark_complete(Ok(())).await;
                }));

                let stale_client = transfer_client.clone();
                let stale_key = stale_key.clone();
                join_handles.push(tokio::spawn(async move {
                    let request = LeaderTransferRequest {
                        key: stale_key,
                        uuid: uuid::Uuid::new_v4(),
                        requirement: None,
                        request_type: RequestType::Immediate,
                        chained: false,
                    };
                    let handle = stale_client.schedule_transfer(request).await.unwrap();
                    handle.mark_complete(Ok(())).await;
                }));
            }
        }

        for _ in 0..expected_messages {
            assert!(scheduler.step().await);
        }

        for handle in join_handles {
            handle.await.unwrap();
        }

        drop(transfer_client);
        drop(scheduler_tx);
        drop(worker_client);

        for (key, binding_counters) in completions {
            for counter in binding_counters {
                assert_eq!(
                    counter.load(Ordering::Acquire),
                    num_immediate_ops as u64,
                    "binding counter mismatch for key {}",
                    key
                );
            }
        }
    }
}
