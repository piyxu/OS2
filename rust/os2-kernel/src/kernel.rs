use std::collections::VecDeque;

use serde::Serialize;

use crate::capability::{
    CapabilityError, CapabilityHandle, CapabilityLimits, CapabilityRegistry, CapabilityUsage,
};
use crate::capability_chain::CapabilityGrantLedger;
use crate::clock::DeterministicClock;
use crate::entropy::EntropyBalancer;
use crate::event_bus::{EventBuilder, EventBus, EventKind, KernelEvent};
use crate::evolution::{EvolverAgent, SelfEvolutionReport, SelfEvolutionTrigger};
use crate::io_queue::{CompletedIoOperation, IoOperationKind, IoOperationRecord, IoQueue};
use crate::memory::{EpisodicRecord, SemanticMemoryBus, ShortTermRecord};
use crate::resource::{
    ResourceError, ResourceGovernor, ResourceRequest, ResourceType, ResourceUsageSnapshot,
};
use crate::resource_monitor::{ResourceObservation, ResourceObserver};
use crate::scheduler::{KernelToken, Scheduler, SchedulerError, TokenId, TokenKind};
use crate::security::{ModuleMetadata, ModuleSecurityError, ModuleSecurityManager};
use crate::snapshot::{
    SnapshotEngine, SnapshotId, SnapshotLedgerIntegrityFailureKind, SnapshotLedgerIntegrityReport,
    SnapshotRollbackError, SnapshotRollbackRecord, SnapshotState,
};
use crate::symbolic::SymbolicLogicEngine;
use crate::telemetry::{TelemetryFrame, TelemetrySynchronizer};
use crate::token_signing::{
    TokenSignature, TokenSignatureLedger, TokenSignatureRecord, TokenSigningError,
    derive_signing_key,
};
use ed25519_dalek::SigningKey;
use thiserror::Error;

pub trait TokenExecutor {
    fn execute(&mut self, token: &KernelToken, ctx: &mut ExecutionContext) -> ExecutionStatus;
}

pub enum ExecutionStatus {
    Completed,
    Yield(KernelToken),
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum SnapshotOperationError {
    #[error(transparent)]
    Engine(#[from] SnapshotRollbackError),
    #[error("memory snapshot {0:?} missing for rollback")]
    MemoryUnavailable(SnapshotId),
}

fn scheduled_payload(token: &KernelToken) -> serde_json::Value {
    serde_json::json!({
        "kind": token.kind,
        "priority": token.priority,
        "cost": token.cost,
        "context_hash": token.context_hash,
        "goal": token.goal,
        "dependencies": token
            .dependencies
            .iter()
            .map(|id| id.raw())
            .collect::<Vec<_>>(),
        "granted_capabilities": token
            .granted_capabilities
            .iter()
            .map(|handle| *handle)
            .collect::<Vec<_>>(),
        "resource_request": token.resources,
    })
}

#[derive(Debug, Clone, Serialize)]
pub struct AsyncScheduledToken {
    pub token_id: u64,
    pub kind: TokenKind,
    pub priority: u8,
    pub goal: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct AsyncStepReport {
    pub description: String,
    pub scheduled: Vec<AsyncScheduledToken>,
}

#[derive(Debug, Clone, Serialize)]
pub struct AsyncBlockReport {
    pub label: String,
    pub steps: Vec<AsyncStepReport>,
}

#[derive(Debug, Clone)]
pub struct AsyncSchedule {
    pub kind: TokenKind,
    pub priority: u8,
    pub cost: u64,
    pub capability: Option<CapabilityHandle>,
    pub payload: serde_json::Value,
    pub context: Option<String>,
    pub goal: Option<String>,
    pub granted_capabilities: Vec<CapabilityHandle>,
    pub resources: Option<ResourceRequest>,
}

impl AsyncSchedule {
    pub fn new(kind: TokenKind, priority: u8, cost: u64) -> Self {
        Self {
            kind,
            priority,
            cost,
            capability: None,
            payload: serde_json::Value::Null,
            context: None,
            goal: None,
            granted_capabilities: Vec::new(),
            resources: None,
        }
    }

    pub fn with_capability(mut self, capability: CapabilityHandle) -> Self {
        self.capability = Some(capability);
        self
    }

    pub fn with_payload(mut self, payload: serde_json::Value) -> Self {
        self.payload = payload;
        self
    }

    pub fn with_context(mut self, context: impl Into<String>) -> Self {
        self.context = Some(context.into());
        self
    }

    pub fn with_goal(mut self, goal: impl Into<String>) -> Self {
        self.goal = Some(goal.into());
        self
    }

    pub fn with_granted_capabilities(
        mut self,
        capabilities: impl Into<Vec<CapabilityHandle>>,
    ) -> Self {
        self.granted_capabilities = capabilities.into();
        self
    }

    pub fn with_resources(mut self, resources: ResourceRequest) -> Self {
        self.resources = Some(resources);
        self
    }
}

#[derive(Debug, Clone)]
pub enum AsyncInstruction {
    Schedule(AsyncSchedule),
}

#[derive(Debug, Clone)]
pub struct AsyncStep {
    pub description: String,
    pub instructions: Vec<AsyncInstruction>,
}

impl AsyncStep {
    pub fn new(description: impl Into<String>) -> Self {
        Self {
            description: description.into(),
            instructions: Vec::new(),
        }
    }

    pub fn schedule(mut self, schedule: AsyncSchedule) -> Self {
        self.instructions.push(AsyncInstruction::Schedule(schedule));
        self
    }

    pub fn push_instruction(&mut self, instruction: AsyncInstruction) {
        self.instructions.push(instruction);
    }
}

#[derive(Debug, Clone)]
pub struct AsyncBlock {
    pub label: String,
    pub steps: Vec<AsyncStep>,
}

impl AsyncBlock {
    pub fn new(label: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            steps: Vec::new(),
        }
    }

    pub fn step(mut self, step: AsyncStep) -> Self {
        self.steps.push(step);
        self
    }

    pub fn push_step(&mut self, step: AsyncStep) {
        self.steps.push(step);
    }
}

#[derive(Debug, Default)]
struct AsyncBlockOrchestrator {
    queue: VecDeque<AsyncBlock>,
    history: Vec<AsyncBlockReport>,
}

impl AsyncBlockOrchestrator {
    fn new() -> Self {
        Self::default()
    }

    fn enqueue(&mut self, block: AsyncBlock) {
        self.queue.push_back(block);
    }

    fn history(&self) -> &[AsyncBlockReport] {
        &self.history
    }

    fn drain(&mut self) -> Vec<AsyncBlock> {
        self.queue.drain(..).collect()
    }

    fn record(&mut self, report: AsyncBlockReport) {
        self.history.push(report);
    }
}

pub struct Kernel {
    capabilities: CapabilityRegistry,
    capability_grants: CapabilityGrantLedger,
    scheduler: Scheduler,
    events: EventBus,
    clock: DeterministicClock,
    snapshots: SnapshotEngine,
    memory: SemanticMemoryBus,
    entropy: EntropyBalancer,
    next_token_id: u64,
    module_security: ModuleSecurityManager,
    resources: ResourceGovernor,
    token_ledger: TokenSignatureLedger,
    token_signer: SigningKey,
    token_signer_id: String,
    telemetry: TelemetrySynchronizer,
    resource_observer: ResourceObserver,
    symbolic_logic: SymbolicLogicEngine,
    async_orchestrator: AsyncBlockOrchestrator,
    self_evolution: SelfEvolutionTrigger,
    io_queue: IoQueue,
}

impl Kernel {
    pub fn new(seed: u64) -> Self {
        let mut module_security = ModuleSecurityManager::new();
        let token_signer = derive_signing_key(seed);
        module_security
            .register_trusted_key("kernel.system", token_signer.verifying_key().as_bytes())
            .expect("kernel signing key registration");
        {
            let policy = module_security.policy_mut();
            policy.require_signatures(true);
            policy.allow_signer_for_prefix("kernel", "kernel.system");
        }

        Self {
            capabilities: CapabilityRegistry::new(),
            capability_grants: CapabilityGrantLedger::new(),
            scheduler: Scheduler::new(),
            events: EventBus::new(),
            clock: DeterministicClock::new(),
            snapshots: SnapshotEngine::new(),
            memory: SemanticMemoryBus::new(),
            entropy: EntropyBalancer::new(seed),
            next_token_id: 0,
            module_security,
            resources: ResourceGovernor::new(),
            token_ledger: TokenSignatureLedger::new(),
            token_signer,
            token_signer_id: "kernel.system".into(),
            telemetry: TelemetrySynchronizer::new(),
            resource_observer: ResourceObserver::new(),
            symbolic_logic: SymbolicLogicEngine::new(),
            async_orchestrator: AsyncBlockOrchestrator::new(),
            self_evolution: SelfEvolutionTrigger::new(EvolverAgent::default()),
            io_queue: IoQueue::new(),
        }
    }

    pub fn register_capability(
        &mut self,
        name: impl Into<String>,
        limits: CapabilityLimits,
    ) -> CapabilityHandle {
        let name_str = name.into();
        let handle = self.capabilities.register(name_str.clone(), limits.clone());
        self.capability_grants
            .append_signed(
                name_str,
                handle,
                limits,
                self.token_signer_id.as_str(),
                &self.token_signer,
                &self.module_security,
            )
            .expect("capability grant signed");
        handle
    }

    pub fn capability_usage(&self, handle: CapabilityHandle) -> Option<CapabilityUsage> {
        self.capabilities
            .info(handle)
            .map(|info| info.usage.clone())
    }

    pub fn module_security(&self) -> &ModuleSecurityManager {
        &self.module_security
    }

    pub fn module_security_mut(&mut self) -> &mut ModuleSecurityManager {
        &mut self.module_security
    }

    pub fn register_signed_module(
        &mut self,
        name: impl Into<String>,
        bytes: &[u8],
        signature: &[u8],
        key_id: &str,
    ) -> Result<ModuleMetadata, ModuleSecurityError> {
        self.module_security
            .register_module(name, bytes, signature, key_id)
    }

    pub fn ensure_module_allowed(&self, hash: &str) -> Result<(), ModuleSecurityError> {
        self.module_security.ensure_allowed(hash)
    }

    pub fn revoke_module(&mut self, hash: &str) {
        self.module_security.revoke_module(hash);
    }

    pub fn configure_resource_limit(&mut self, resource: ResourceType, limit: Option<u64>) {
        self.resources.set_limit(resource, limit);
    }

    pub fn resource_usage(&self) -> ResourceUsageSnapshot {
        self.resources.usage()
    }

    pub fn telemetry_frames(&self) -> &[TelemetryFrame] {
        self.telemetry.frames()
    }

    pub fn resource_observations(&self) -> &[ResourceObservation] {
        self.resource_observer.history()
    }

    pub fn symbolic_logic(&self) -> &SymbolicLogicEngine {
        &self.symbolic_logic
    }

    pub fn symbolic_logic_mut(&mut self) -> &mut SymbolicLogicEngine {
        &mut self.symbolic_logic
    }

    pub fn queue_async_block(&mut self, block: AsyncBlock) {
        self.async_orchestrator.enqueue(block);
    }

    pub fn run_boot_orchestrator(&mut self) {
        let blocks = self.async_orchestrator.drain();
        for block in blocks {
            let report = self.execute_async_block(&block);
            self.async_orchestrator.record(report.clone());
            self.publish_async_report(&report);
        }
    }

    pub fn async_boot_history(&self) -> &[AsyncBlockReport] {
        self.async_orchestrator.history()
    }

    pub fn take_self_evolution_reports(&mut self) -> Vec<SelfEvolutionReport> {
        self.self_evolution.take_pending()
    }

    pub fn consume_capability(
        &mut self,
        handle: CapabilityHandle,
        tokens: u64,
    ) -> Result<CapabilityUsage, CapabilityError> {
        self.capabilities.consume(handle, tokens)
    }

    pub fn allocate_token(&mut self, priority: u8, cost: u64, kind: TokenKind) -> KernelToken {
        let id = TokenId::new(self.next_token_id);
        self.next_token_id += 1;
        KernelToken::new(id, priority, cost, kind)
    }

    pub fn submit_token(&mut self, mut token: KernelToken) {
        let prev_hash = self.token_ledger.head_hash().map(|s| s.to_string());
        if token.signature.is_none() {
            let signature = TokenSignature::sign(
                &token,
                self.token_signer_id.as_str(),
                &self.token_signer,
                prev_hash.as_deref(),
            );
            token.signature = Some(signature);
        }

        match self
            .token_ledger
            .verify_and_append(&token, &self.module_security)
        {
            Ok(record) => {
                self.emit_system_call_event(&token, &record);
            }
            Err(err) => {
                self.events.publish(
                    &mut self.clock,
                    EventBuilder::new(token.id, EventKind::SecurityViolation)
                        .detail(token_signing_error_payload(&err)),
                );
                return;
            }
        }

        let event =
            EventBuilder::new(token.id, EventKind::Scheduled).detail(scheduled_payload(&token));
        self.events.publish(&mut self.clock, event);
        self.scheduler.schedule(token);
    }

    pub fn token_chain_head(&self) -> Option<&TokenSignatureRecord> {
        self.token_ledger.records().last()
    }

    pub fn token_chain_head_hash(&self) -> Option<&str> {
        self.token_ledger.head_hash()
    }

    pub fn queue_io_operation(
        &mut self,
        token_id: TokenId,
        kind: IoOperationKind,
        detail: impl Serialize,
    ) -> IoOperationRecord {
        let queued_at = self.clock.now();
        let record = self.io_queue.enqueue(token_id, kind, detail, queued_at);
        self.events.publish(
            &mut self.clock,
            EventBuilder::new(token_id, EventKind::IoQueued).detail(record.to_json()),
        );
        record
    }

    pub fn complete_next_io_operation(&mut self) -> Option<CompletedIoOperation> {
        let completed_at = self.clock.now();
        let completion = self.io_queue.complete_next(completed_at);
        if let Some(ref record) = completion {
            self.events.publish(
                &mut self.clock,
                EventBuilder::new(record.token_id, EventKind::IoCompleted).detail(record.to_json()),
            );
        }
        completion
    }

    pub fn pending_io_operations(&self) -> Vec<IoOperationRecord> {
        self.io_queue.iter().cloned().collect()
    }

    fn emit_system_call_event(&mut self, token: &KernelToken, record: &TokenSignatureRecord) {
        let event = EventBuilder::new(token.id, EventKind::Custom("system_call".into()))
            .detail(token_ledger_payload(record));
        self.events.publish(&mut self.clock, event);
    }

    fn execute_async_block(&mut self, block: &AsyncBlock) -> AsyncBlockReport {
        let mut step_reports = Vec::new();
        for (index, step) in block.steps.iter().enumerate() {
            let mut scheduled = Vec::new();
            for instruction in &step.instructions {
                let AsyncInstruction::Schedule(schedule) = instruction;
                let mut token =
                    self.allocate_token(schedule.priority, schedule.cost, schedule.kind);
                if let Some(handle) = schedule.capability {
                    token = token.with_capability(handle);
                }
                if let Some(goal) = &schedule.goal {
                    token = token.with_goal(goal.clone());
                }
                let context = schedule
                    .context
                    .clone()
                    .unwrap_or_else(|| format!("async::{}::{}", block.label, index));
                token = token.with_context_hash(context);
                if !schedule.granted_capabilities.is_empty() {
                    token = token.with_granted_capabilities(schedule.granted_capabilities.clone());
                }
                if let Some(resources) = schedule.resources {
                    token = token.with_resource_request(resources);
                }
                token = token.with_payload(schedule.payload.clone());
                let cloned = token.clone();
                self.submit_token(token);
                scheduled.push(AsyncScheduledToken {
                    token_id: cloned.id.raw(),
                    kind: cloned.kind,
                    priority: cloned.priority,
                    goal: cloned.goal.clone(),
                });
            }
            step_reports.push(AsyncStepReport {
                description: step.description.clone(),
                scheduled,
            });
        }

        AsyncBlockReport {
            label: block.label.clone(),
            steps: step_reports,
        }
    }

    fn publish_async_report(&mut self, report: &AsyncBlockReport) {
        let detail = serde_json::to_value(report).unwrap_or(serde_json::Value::Null);
        let event = EventBuilder::new(TokenId::new(0), EventKind::Custom("async_boot".into()))
            .detail(detail);
        self.events.publish(&mut self.clock, event);
    }

    fn observe_resources(&mut self) {
        let usage = self.resources.usage();
        let timestamp = self.clock.now();
        self.resource_observer.record(timestamp, usage);
    }

    pub fn process_next<E>(&mut self, executor: &mut E) -> bool
    where
        E: TokenExecutor,
    {
        let token = match self.scheduler.next() {
            Some(token) => token,
            None => return false,
        };

        if let Err(err) = self.scheduler.start(token.id, token.cost) {
            self.events.publish(
                &mut self.clock,
                EventBuilder::new(token.id, EventKind::BudgetViolation)
                    .detail(scheduler_error_payload(err)),
            );
            self.scheduler.complete(token.id);
            self.observe_resources();
            return true;
        }

        let capability_detail = if let Some(handle) = token.capability {
            let grant_record = match self.capability_grants.ensure_signed(handle) {
                Ok(record) => record,
                Err(_) => {
                    self.events.publish(
                        &mut self.clock,
                        EventBuilder::new(token.id, EventKind::CapabilityViolation).detail(
                            serde_json::json!({
                                "error": "unsigned_capability",
                                "capability_handle": handle.raw(),
                            }),
                        ),
                    );
                    self.scheduler.complete(token.id);
                    self.observe_resources();
                    return true;
                }
            };

            match self.capabilities.consume(handle, token.cost) {
                Ok(usage) => {
                    let mut map = serde_json::Map::new();
                    if let Some(info) = self.capabilities.info(handle) {
                        map.insert(
                            "capability_handle".into(),
                            serde_json::json!(info.handle.raw()),
                        );
                        map.insert(
                            "capability_name".into(),
                            serde_json::json!(info.name.to_string()),
                        );
                    } else {
                        map.insert("capability_handle".into(), serde_json::json!(handle.raw()));
                    }
                    map.insert(
                        "capability_grant".into(),
                        serde_json::json!({
                            "entry_hash": grant_record.entry_hash,
                            "signer": grant_record.signer,
                        }),
                    );
                    map.insert("invocations".into(), serde_json::json!(usage.invocations));
                    map.insert("tokens_total".into(), serde_json::json!(usage.tokens));
                    map.insert("tokens_consumed".into(), serde_json::json!(token.cost));
                    Some(map)
                }
                Err(err) => {
                    self.events.publish(
                        &mut self.clock,
                        EventBuilder::new(token.id, EventKind::CapabilityViolation)
                            .detail(error_payload(err)),
                    );
                    self.scheduler.complete(token.id);
                    self.observe_resources();
                    return true;
                }
            }
        } else {
            None
        };

        let resource_usage = match self.resources.consume(&token.resources) {
            Ok(usage) => usage,
            Err(err) => {
                self.events.publish(
                    &mut self.clock,
                    EventBuilder::new(token.id, EventKind::ResourceViolation)
                        .detail(resource_error_payload(err)),
                );
                self.scheduler.complete(token.id);
                self.observe_resources();
                return true;
            }
        };

        if let Ok(encoded_usage) = serde_json::to_vec(&resource_usage) {
            self.entropy.mix_entropy(&encoded_usage);
        }

        let mut detail = serde_json::json!({
            "resources": {
                "requested": token.resources,
                "usage": resource_usage,
            }
        });

        if let Some(map) = capability_detail {
            if let serde_json::Value::Object(ref mut detail_map) = detail {
                detail_map.extend(map);
            }
        }

        self.events.publish(
            &mut self.clock,
            EventBuilder::new(token.id, EventKind::Started).detail(detail),
        );

        let signer_id = self.token_signer_id.as_str();
        let mut ctx = ExecutionContext {
            capability_registry: &mut self.capabilities,
            scheduler: &mut self.scheduler,
            events: &mut self.events,
            snapshots: &mut self.snapshots,
            clock: &mut self.clock,
            memory: &mut self.memory,
            resource_governor: &mut self.resources,
            entropy: &mut self.entropy,
            token_ledger: &mut self.token_ledger,
            module_security: &self.module_security,
            token_signer: &self.token_signer,
            token_signer_id: signer_id,
            telemetry: &mut self.telemetry,
            symbolic_logic: &mut self.symbolic_logic,
            async_orchestrator: &mut self.async_orchestrator,
            io_queue: &mut self.io_queue,
        };

        match executor.execute(&token, &mut ctx) {
            ExecutionStatus::Completed => {
                ctx.emit(EventBuilder::new(token.id, EventKind::Completed));
            }
            ExecutionStatus::Yield(next) => {
                ctx.emit(EventBuilder::new(token.id, EventKind::Yielded));
                ctx.spawn(next);
            }
        }

        drop(ctx);
        self.scheduler.complete(token.id);
        self.observe_resources();
        true
    }

    pub fn process_until_idle<E>(&mut self, executor: &mut E)
    where
        E: TokenExecutor,
    {
        while self.process_next(executor) {}
    }

    pub fn drain_events(&mut self) -> Vec<KernelEvent> {
        let mut events = self.events.drain();
        if let Some(report) = self.self_evolution.observe(&events) {
            let detail = serde_json::to_value(&report).unwrap_or(serde_json::Value::Null);
            if report.triggered {
                let alert_timestamp = self.clock.tick();
                events.push(KernelEvent {
                    token_id: TokenId::new(0),
                    kind: EventKind::PolicyAlert,
                    detail: detail.clone(),
                    timestamp: alert_timestamp,
                });
            }
            let evolution_timestamp = self.clock.tick();
            events.push(KernelEvent {
                token_id: TokenId::new(0),
                kind: EventKind::Custom("self_evolution".into()),
                detail,
                timestamp: evolution_timestamp,
            });
        }
        events
    }

    pub fn checkpoint(&mut self, label: impl Into<String>, state: serde_json::Value) -> SnapshotId {
        let snapshot = self.snapshots.checkpoint(label, state);
        self.memory.snapshot(snapshot);
        self.events.publish(
            &mut self.clock,
            EventBuilder::new(TokenId::new(0), EventKind::Checkpoint)
                .detail(serde_json::json!({"snapshot": snapshot.raw()})),
        );
        if let Some(entry) = self.snapshots.ledger_head() {
            let usage = self.resources.usage();
            let token_head = self.token_ledger.head_hash().map(|hash| hash.to_string());
            self.telemetry.record(entry, usage, token_head);
        }
        let _ = self.snapshots.check_integrity();
        snapshot
    }

    pub fn rollback_snapshot(
        &mut self,
        target: SnapshotId,
    ) -> Result<SnapshotRollbackRecord, SnapshotOperationError> {
        if !self.memory.restore(target) {
            return Err(SnapshotOperationError::MemoryUnavailable(target));
        }

        match self.snapshots.rollback(target) {
            Ok(record) => {
                self.memory.snapshot(record.committed_snapshot);
                if let Some(entry) = self.snapshots.ledger_head() {
                    let usage = self.resources.usage();
                    let token_head = self.token_ledger.head_hash().map(|hash| hash.to_string());
                    self.telemetry.record(entry, usage, token_head);
                }
                let _ = self.snapshots.check_integrity();
                Ok(record)
            }
            Err(err) => Err(SnapshotOperationError::Engine(err)),
        }
    }

    pub fn restore(&self, id: SnapshotId) -> Option<&SnapshotState> {
        self.snapshots.restore(id)
    }

    pub fn latest_snapshot(&self) -> Option<(&SnapshotId, &SnapshotState)> {
        self.snapshots.latest()
    }

    pub fn last_snapshot_integrity(&self) -> Option<&SnapshotLedgerIntegrityReport> {
        self.snapshots.last_integrity()
    }

    pub fn memory(&self) -> &SemanticMemoryBus {
        &self.memory
    }

    pub fn restore_memory(&mut self, id: SnapshotId) -> bool {
        self.memory.restore(id)
    }
}

fn error_payload(error: CapabilityError) -> serde_json::Value {
    match error {
        CapabilityError::NotFound => serde_json::json!({"error": "not_found"}),
        CapabilityError::InvocationBudgetExceeded => {
            serde_json::json!({"error": "invocation_budget_exceeded"})
        }
        CapabilityError::TokenBudgetExceeded => {
            serde_json::json!({"error": "token_budget_exceeded"})
        }
    }
}

fn resource_error_payload(error: ResourceError) -> serde_json::Value {
    match error {
        ResourceError::BudgetExceeded {
            resource,
            attempted,
            remaining,
        } => serde_json::json!({
            "error": "resource_budget_exceeded",
            "resource": resource.as_str(),
            "attempted": attempted,
            "remaining": remaining,
        }),
    }
}

fn scheduler_error_payload(error: SchedulerError) -> serde_json::Value {
    match error {
        SchedulerError::BudgetExceeded {
            attempted,
            remaining,
            ..
        } => serde_json::json!({
            "error": "budget_exceeded",
            "attempted": attempted,
            "remaining": remaining,
        }),
        SchedulerError::MissingBudget { .. } => {
            serde_json::json!({"error": "missing_budget"})
        }
    }
}

fn token_signing_error_payload(error: &TokenSigningError) -> serde_json::Value {
    match error {
        TokenSigningError::MissingSignature(token_id) => serde_json::json!({
            "error": "missing_signature",
            "token_id": token_id.raw(),
        }),
        TokenSigningError::UnknownSigner { signer, token_id } => {
            serde_json::json!({
                "error": "unknown_signer",
                "signer": signer,
                "token_id": token_id.raw(),
            })
        }
        TokenSigningError::InvalidSignature { token_id } => serde_json::json!({
            "error": "invalid_signature",
            "token_id": token_id.raw(),
        }),
        TokenSigningError::PayloadMismatch { token_id } => serde_json::json!({
            "error": "payload_mismatch",
            "token_id": token_id.raw(),
        }),
        TokenSigningError::ChainMismatch { token_id } => serde_json::json!({
            "error": "chain_mismatch",
            "token_id": token_id.raw(),
        }),
    }
}

fn token_ledger_payload(record: &TokenSignatureRecord) -> serde_json::Value {
    serde_json::json!({
        "signer": record.signer,
        "payload_hash": record.payload_hash,
        "previous_hash": record.previous_hash,
        "entry_hash": record.entry_hash,
    })
}

fn snapshot_integrity_payload(report: &SnapshotLedgerIntegrityReport) -> serde_json::Value {
    let mut detail = serde_json::Map::new();
    detail.insert("is_valid".into(), serde_json::Value::Bool(report.is_valid));
    detail.insert(
        "entries_checked".into(),
        serde_json::json!(report.checked_entries),
    );
    if let Some(hash) = &report.head_hash {
        detail.insert("head_hash".into(), serde_json::json!(hash));
    }

    match &report.failure {
        Some(failure) => {
            detail.insert("status".into(), serde_json::json!("invalid"));
            detail.insert(
                "failure_kind".into(),
                serde_json::json!(failure.kind.as_str()),
            );
            detail.insert(
                "entry_snapshot".into(),
                serde_json::json!(failure.entry.snapshot_id),
            );
            detail.insert(
                "entry_hash".into(),
                serde_json::json!(failure.entry.entry_hash.clone()),
            );
            detail.insert(
                "state_hash".into(),
                serde_json::json!(failure.entry.state_hash.clone()),
            );
            detail.insert(
                "label".into(),
                serde_json::json!(failure.entry.label.clone()),
            );
            detail.insert(
                "timestamp".into(),
                serde_json::json!(failure.entry.timestamp),
            );
            if let Some(prev) = &failure.entry.previous_hash {
                detail.insert(
                    "observed_previous_hash".into(),
                    serde_json::json!(prev.clone()),
                );
            }
            match failure.kind {
                SnapshotLedgerIntegrityFailureKind::EntryHashMismatch => {
                    if let Some(expected) = &failure.expected_hash {
                        detail.insert("expected_hash".into(), serde_json::json!(expected.clone()));
                    }
                }
                SnapshotLedgerIntegrityFailureKind::PreviousHashMismatch => {
                    if let Some(expected_prev) = &failure.expected_previous {
                        detail.insert(
                            "expected_previous_hash".into(),
                            serde_json::json!(expected_prev.clone()),
                        );
                    }
                }
            }
        }
        None => {
            detail.insert("status".into(), serde_json::json!("valid"));
        }
    }

    serde_json::Value::Object(detail)
}

pub struct ExecutionContext<'a> {
    capability_registry: &'a mut CapabilityRegistry,
    scheduler: &'a mut Scheduler,
    events: &'a mut EventBus,
    snapshots: &'a mut SnapshotEngine,
    clock: &'a mut DeterministicClock,
    memory: &'a mut SemanticMemoryBus,
    resource_governor: &'a mut ResourceGovernor,
    entropy: &'a mut EntropyBalancer,
    token_ledger: &'a mut TokenSignatureLedger,
    module_security: &'a ModuleSecurityManager,
    token_signer: &'a SigningKey,
    token_signer_id: &'a str,
    telemetry: &'a mut TelemetrySynchronizer,
    symbolic_logic: &'a mut SymbolicLogicEngine,
    async_orchestrator: &'a mut AsyncBlockOrchestrator,
    io_queue: &'a mut IoQueue,
}

impl<'a> ExecutionContext<'a> {
    pub fn emit(&mut self, event: EventBuilder) {
        self.events.publish(self.clock, event);
    }

    pub fn emit_custom(
        &mut self,
        token_id: TokenId,
        label: impl Into<String>,
        detail: impl Serialize,
    ) {
        let event = EventBuilder::new(token_id, EventKind::Custom(label.into())).detail(detail);
        self.emit(event);
    }

    pub fn spawn(&mut self, mut token: KernelToken) {
        let prev_hash = self.token_ledger.head_hash().map(|s| s.to_string());
        if token.signature.is_none() {
            let signature = TokenSignature::sign(
                &token,
                self.token_signer_id,
                self.token_signer,
                prev_hash.as_deref(),
            );
            token.signature = Some(signature);
        }

        match self
            .token_ledger
            .verify_and_append(&token, self.module_security)
        {
            Ok(record) => {
                let event = EventBuilder::new(token.id, EventKind::Custom("system_call".into()))
                    .detail(token_ledger_payload(&record));
                self.emit(event);
            }
            Err(err) => {
                self.emit(
                    EventBuilder::new(token.id, EventKind::SecurityViolation)
                        .detail(token_signing_error_payload(&err)),
                );
                return;
            }
        }

        let event =
            EventBuilder::new(token.id, EventKind::Scheduled).detail(scheduled_payload(&token));
        self.emit(event);
        self.scheduler.schedule(token);
    }

    pub fn checkpoint(
        &mut self,
        token_id: TokenId,
        label: impl Into<String>,
        state: serde_json::Value,
    ) -> SnapshotId {
        let snapshot = self.snapshots.checkpoint(label, state);
        self.memory.snapshot(snapshot);
        self.emit(
            EventBuilder::new(token_id, EventKind::SnapshotCreated)
                .detail(serde_json::json!({"snapshot": snapshot.raw()})),
        );
        self.emit(
            EventBuilder::new(token_id, EventKind::Checkpoint)
                .detail(serde_json::json!({"snapshot": snapshot.raw()})),
        );
        if let Some(entry) = self.snapshots.ledger_head() {
            let usage = self.resource_governor.usage();
            let token_head = self.token_ledger.head_hash().map(|hash| hash.to_string());
            self.telemetry.record(entry, usage, token_head);
        }
        let report = self.snapshots.check_integrity();
        let kind = if report.is_valid {
            EventKind::SnapshotIntegrityVerified
        } else {
            EventKind::SnapshotIntegrityViolation
        };
        self.emit(EventBuilder::new(token_id, kind).detail(snapshot_integrity_payload(&report)));
        snapshot
    }

    pub fn rollback_snapshot(
        &mut self,
        token_id: TokenId,
        target: SnapshotId,
    ) -> Result<SnapshotRollbackRecord, SnapshotOperationError> {
        let from_snapshot = self.snapshots.active_snapshot();
        self.emit(
            EventBuilder::new(token_id, EventKind::SnapshotRollbackStarted).detail(
                serde_json::json!({
                    "from": from_snapshot.map(|id| id.raw()),
                    "to": target.raw(),
                }),
            ),
        );

        if !self.memory.restore(target) {
            self.emit(
                EventBuilder::new(token_id, EventKind::SnapshotRollbackFailed).detail(
                    serde_json::json!({
                        "from": from_snapshot.map(|id| id.raw()),
                        "to": target.raw(),
                        "error": "memory_missing",
                    }),
                ),
            );
            return Err(SnapshotOperationError::MemoryUnavailable(target));
        }

        match self.snapshots.rollback(target) {
            Ok(record) => {
                self.memory.snapshot(record.committed_snapshot);
                let detail = serde_json::json!({
                    "from": record.from_snapshot.map(|id| id.raw()),
                    "to": record.to_snapshot.raw(),
                    "committed": record.committed_snapshot.raw(),
                    "entry_hash": record.entry_hash.clone(),
                });
                self.emit(
                    EventBuilder::new(token_id, EventKind::SnapshotRollbackCommitted)
                        .detail(detail),
                );
                if let Some(entry) = self.snapshots.ledger_head() {
                    let usage = self.resource_governor.usage();
                    let token_head = self.token_ledger.head_hash().map(|hash| hash.to_string());
                    self.telemetry.record(entry, usage, token_head);
                }
                let report = self.snapshots.check_integrity();
                let kind = if report.is_valid {
                    EventKind::SnapshotIntegrityVerified
                } else {
                    EventKind::SnapshotIntegrityViolation
                };
                self.emit(
                    EventBuilder::new(token_id, kind).detail(snapshot_integrity_payload(&report)),
                );
                Ok(record)
            }
            Err(err) => {
                self.emit(
                    EventBuilder::new(token_id, EventKind::SnapshotRollbackFailed).detail(
                        serde_json::json!({
                            "from": from_snapshot.map(|id| id.raw()),
                            "to": target.raw(),
                            "error": err.to_string(),
                        }),
                    ),
                );
                Err(SnapshotOperationError::Engine(err))
            }
        }
    }

    pub fn symbolic_logic(&mut self) -> &mut SymbolicLogicEngine {
        self.symbolic_logic
    }

    pub fn queue_async_block(&mut self, block: AsyncBlock) {
        self.async_orchestrator.enqueue(block);
    }

    pub fn queue_io_operation(
        &mut self,
        token_id: TokenId,
        kind: IoOperationKind,
        detail: impl Serialize,
    ) -> IoOperationRecord {
        let queued_at = self.clock.now();
        let record = self.io_queue.enqueue(token_id, kind, detail, queued_at);
        self.emit(EventBuilder::new(token_id, EventKind::IoQueued).detail(record.to_json()));
        record
    }

    pub fn complete_next_io_operation(&mut self) -> Option<CompletedIoOperation> {
        let completed_at = self.clock.now();
        let completion = self.io_queue.complete_next(completed_at);
        if let Some(ref record) = completion {
            self.emit(
                EventBuilder::new(record.token_id, EventKind::IoCompleted).detail(record.to_json()),
            );
        }
        completion
    }

    pub fn pending_io_operations(&mut self) -> Vec<IoOperationRecord> {
        self.io_queue.iter().cloned().collect()
    }

    pub fn capability_usage(&self, handle: CapabilityHandle) -> Option<CapabilityUsage> {
        self.capability_registry
            .info(handle)
            .map(|info| info.usage.clone())
    }

    pub fn random_u64(&mut self) -> u64 {
        self.entropy.next_u64()
    }

    pub fn resource_usage(&self) -> ResourceUsageSnapshot {
        self.resource_governor.usage()
    }

    pub fn write_short_term_memory(
        &mut self,
        token_id: TokenId,
        context_hash: impl Into<String>,
        value: serde_json::Value,
    ) {
        let context_hash = context_hash.into();
        let timestamp = self.clock.now();
        let record = ShortTermRecord {
            context_hash: context_hash.clone(),
            value: value.clone(),
            timestamp,
        };
        self.memory.record_short_term(record);
        self.emit(
            EventBuilder::new(token_id, EventKind::MemoryWrite).detail(serde_json::json!({
                "store": "short_term",
                "context_hash": context_hash,
            })),
        );
    }

    pub fn upsert_long_term_memory(
        &mut self,
        token_id: TokenId,
        key: impl Into<String>,
        value: serde_json::Value,
    ) {
        let key = key.into();
        self.memory.upsert_long_term(key.clone(), value);
        self.emit(
            EventBuilder::new(token_id, EventKind::MemoryWrite).detail(serde_json::json!({
                "store": "long_term",
                "key": key,
            })),
        );
    }

    pub fn append_episodic_memory(&mut self, token_id: TokenId, detail: serde_json::Value) {
        let record = EpisodicRecord {
            token_id,
            detail: detail.clone(),
            timestamp: self.clock.now(),
        };
        self.memory.append_episodic(record);
        self.emit(
            EventBuilder::new(token_id, EventKind::MemoryWrite).detail(serde_json::json!({
                "store": "episodic",
            })),
        );
    }

    pub fn read_long_term_memory(
        &mut self,
        token_id: TokenId,
        key: &str,
    ) -> Option<serde_json::Value> {
        let value = self
            .memory
            .find_long_term(key)
            .map(|record| record.value.clone());
        if value.is_some() {
            self.emit(EventBuilder::new(token_id, EventKind::MemoryRead).detail(
                serde_json::json!({
                    "store": "long_term",
                    "key": key,
                }),
            ));
        }
        value
    }

    pub fn read_short_term_memory(
        &mut self,
        token_id: TokenId,
        context_hash: &str,
    ) -> Option<serde_json::Value> {
        let value = self
            .memory
            .find_latest_short_term(context_hash)
            .map(|record| record.value.clone());
        if value.is_some() {
            self.emit(EventBuilder::new(token_id, EventKind::MemoryRead).detail(
                serde_json::json!({
                    "store": "short_term",
                    "context_hash": context_hash,
                }),
            ));
        }
        value
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io_queue::IoOperationKind;

    struct RecordingExecutor {
        log: Vec<String>,
    }

    impl RecordingExecutor {
        fn new() -> Self {
            Self { log: Vec::new() }
        }
    }

    impl TokenExecutor for RecordingExecutor {
        fn execute(&mut self, token: &KernelToken, ctx: &mut ExecutionContext) -> ExecutionStatus {
            self.log.push(format!("executed:{}", token.id.raw()));
            ctx.emit_custom(
                token.id,
                "execution",
                serde_json::json!({"payload": token.payload}),
            );

            if token.priority > 5 {
                if let Some(handle) = token.capability {
                    let mut spawned = ctx
                        .capability_usage(handle)
                        .map(|usage| usage.invocations)
                        .unwrap_or(0);
                    spawned += 1;
                    let next_token = KernelToken::new(
                        TokenId::new(100 + spawned),
                        token.priority - 1,
                        1,
                        TokenKind::Reflect,
                    )
                    .with_capability(handle);
                    ctx.spawn(next_token);
                }
            }

            ExecutionStatus::Completed
        }
    }

    #[test]
    fn kernel_processes_tokens_and_records_events() {
        let mut kernel = Kernel::new(42);
        let capability = kernel.register_capability(
            "reason",
            CapabilityLimits {
                max_invocations: Some(10),
                max_tokens: Some(100),
            },
        );

        let mut executor = RecordingExecutor::new();
        let token = kernel
            .allocate_token(9, 5, TokenKind::Reason)
            .with_capability(capability)
            .with_payload(serde_json::json!({"goal": "plan"}));

        kernel.submit_token(token);
        kernel.process_until_idle(&mut executor);

        let events = kernel.drain_events();
        assert!(!events.is_empty());
        assert!(
            events
                .iter()
                .any(|event| matches!(event.kind, EventKind::Completed))
        );
        assert!(executor.log.contains(&"executed:0".to_string()));
        assert!(!kernel.resource_observations().is_empty());
    }

    #[test]
    fn capability_violations_are_emitted() {
        let mut kernel = Kernel::new(0);
        let capability = kernel.register_capability(
            "limited",
            CapabilityLimits {
                max_invocations: Some(1),
                max_tokens: Some(1),
            },
        );
        let mut executor = RecordingExecutor::new();
        let token = kernel
            .allocate_token(1, 1, TokenKind::Reason)
            .with_capability(capability);
        let second = kernel
            .allocate_token(1, 10, TokenKind::Reason)
            .with_capability(capability);

        kernel.submit_token(token);
        kernel.submit_token(second);
        kernel.process_until_idle(&mut executor);

        let events = kernel.drain_events();
        assert!(
            events
                .iter()
                .any(|event| matches!(event.kind, EventKind::CapabilityViolation))
        );
    }

    #[test]
    fn checkpointing_creates_snapshot_records() {
        let mut kernel = Kernel::new(0);
        let token = kernel.allocate_token(1, 0, TokenKind::Plan);
        kernel.submit_token(token);

        struct SnapshotExecutor;
        impl TokenExecutor for SnapshotExecutor {
            fn execute(
                &mut self,
                token: &KernelToken,
                ctx: &mut ExecutionContext,
            ) -> ExecutionStatus {
                let snapshot = ctx.checkpoint(
                    token.id,
                    "state",
                    serde_json::json!({"token": token.id.raw()}),
                );
                ctx.emit_custom(
                    token.id,
                    "checkpoint",
                    serde_json::json!({"snapshot": snapshot.raw()}),
                );
                ExecutionStatus::Completed
            }
        }

        let mut executor = SnapshotExecutor;
        kernel.process_until_idle(&mut executor);
        let events = kernel.drain_events();
        assert!(
            events
                .iter()
                .any(|event| matches!(event.kind, EventKind::SnapshotCreated))
        );
        assert!(
            events
                .iter()
                .any(|event| matches!(event.kind, EventKind::SnapshotIntegrityVerified))
        );
        assert!(kernel.latest_snapshot().is_some());
        assert!(!kernel.telemetry_frames().is_empty());
    }

    #[test]
    fn memory_operations_emit_events_and_restore() {
        let mut kernel = Kernel::new(5);
        struct MemoryExecutor;
        impl TokenExecutor for MemoryExecutor {
            fn execute(
                &mut self,
                token: &KernelToken,
                ctx: &mut ExecutionContext,
            ) -> ExecutionStatus {
                ctx.write_short_term_memory(
                    token.id,
                    token.context_hash.clone(),
                    serde_json::json!({"value": 42}),
                );
                ctx.upsert_long_term_memory(
                    token.id,
                    "plan_key",
                    serde_json::json!({"steps": ["one"]}),
                );
                ctx.append_episodic_memory(token.id, serde_json::json!({"event": "acted"}));
                ExecutionStatus::Completed
            }
        }

        let token = kernel
            .allocate_token(5, 1, TokenKind::Reason)
            .with_context_hash("ctx:reason".to_string())
            .with_goal("reason".to_string());
        kernel.submit_token(token);
        kernel.process_until_idle(&mut MemoryExecutor);

        let baseline_short = kernel.memory().short_term_records().len();
        assert!(baseline_short > 0);

        let snapshot = kernel.checkpoint("memory", serde_json::json!({}));
        let events = kernel.drain_events();
        assert!(
            events
                .iter()
                .any(|event| matches!(event.kind, EventKind::MemoryWrite))
        );

        let token = kernel
            .allocate_token(5, 1, TokenKind::Reflect)
            .with_context_hash("ctx:reflect".to_string())
            .with_goal("reflect".to_string());
        kernel.submit_token(token);
        kernel.process_until_idle(&mut MemoryExecutor);
        assert!(kernel.memory().short_term_records().len() > baseline_short);

        assert!(kernel.restore_memory(snapshot));
        assert_eq!(kernel.memory().short_term_records().len(), baseline_short);
    }

    #[test]
    fn io_operations_emit_sequential_events() {
        struct IoExecutor;
        impl TokenExecutor for IoExecutor {
            fn execute(
                &mut self,
                token: &KernelToken,
                ctx: &mut ExecutionContext,
            ) -> ExecutionStatus {
                let first = ctx.queue_io_operation(
                    token.id,
                    IoOperationKind::Open,
                    serde_json::json!({"path": "/var/data"}),
                );
                assert_eq!(first.sequence, 0);

                let second = ctx.queue_io_operation(
                    token.id,
                    IoOperationKind::Write,
                    serde_json::json!({"bytes": 512}),
                );
                assert_eq!(second.sequence, 1);

                let pending = ctx.pending_io_operations();
                assert_eq!(pending.len(), 2);

                ctx.complete_next_io_operation();
                ctx.complete_next_io_operation();
                assert!(ctx.complete_next_io_operation().is_none());

                ExecutionStatus::Completed
            }
        }

        let mut kernel = Kernel::new(0);
        let token = kernel.allocate_token(1, 1, TokenKind::Plan);
        kernel.submit_token(token);
        kernel.process_until_idle(&mut IoExecutor);

        let events = kernel.drain_events();
        let queued = events
            .iter()
            .filter(|event| matches!(event.kind, EventKind::IoQueued))
            .count();
        let completed = events
            .iter()
            .filter(|event| matches!(event.kind, EventKind::IoCompleted))
            .count();

        assert_eq!(queued, 2);
        assert_eq!(completed, 2);
        assert!(kernel.pending_io_operations().is_empty());
    }

    #[test]
    fn snapshot_rollback_emits_atomic_events() {
        let mut kernel = Kernel::new(0);
        let token_a = kernel.allocate_token(5, 1, TokenKind::Plan);
        let token_b = kernel.allocate_token(4, 1, TokenKind::Plan);
        kernel.submit_token(token_a);
        kernel.submit_token(token_b);

        struct RollbackExecutor {
            stage: u8,
            baseline: Option<SnapshotId>,
        }

        impl RollbackExecutor {
            fn new() -> Self {
                Self {
                    stage: 0,
                    baseline: None,
                }
            }
        }

        impl TokenExecutor for RollbackExecutor {
            fn execute(
                &mut self,
                token: &KernelToken,
                ctx: &mut ExecutionContext,
            ) -> ExecutionStatus {
                match self.stage {
                    0 => {
                        let baseline = ctx.checkpoint(
                            token.id,
                            "baseline",
                            serde_json::json!({"value": "baseline"}),
                        );
                        ctx.checkpoint(
                            token.id,
                            "updated",
                            serde_json::json!({"value": "updated"}),
                        );
                        self.baseline = Some(baseline);
                        self.stage = 1;
                    }
                    _ => {
                        let target = self.baseline.expect("baseline available");
                        let record = ctx
                            .rollback_snapshot(token.id, target)
                            .expect("rollback succeeds");
                        assert_ne!(record.committed_snapshot, target);
                    }
                }
                ExecutionStatus::Completed
            }
        }

        let mut executor = RollbackExecutor::new();
        kernel.process_until_idle(&mut executor);

        let events = kernel.drain_events();
        let started = events
            .iter()
            .filter(|event| matches!(event.kind, EventKind::SnapshotRollbackStarted))
            .count();
        let committed = events
            .iter()
            .filter(|event| matches!(event.kind, EventKind::SnapshotRollbackCommitted))
            .count();
        assert_eq!(started, 1);
        assert_eq!(committed, 1);
        assert!(
            events
                .iter()
                .all(|event| !matches!(event.kind, EventKind::SnapshotRollbackFailed))
        );
        assert!(
            events
                .iter()
                .any(|event| matches!(event.kind, EventKind::SnapshotIntegrityVerified))
        );

        let baseline = executor.baseline.expect("baseline captured");
        let records = kernel.snapshots.rollback_records();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].to_snapshot, baseline);
    }

    #[test]
    fn integrity_violation_emits_event() {
        let mut kernel = Kernel::new(0);
        let token = kernel.allocate_token(1, 0, TokenKind::Plan);
        kernel.submit_token(token);

        struct CheckpointOnce;
        impl TokenExecutor for CheckpointOnce {
            fn execute(
                &mut self,
                token: &KernelToken,
                ctx: &mut ExecutionContext,
            ) -> ExecutionStatus {
                ctx.checkpoint(
                    token.id,
                    "baseline",
                    serde_json::json!({"value": "baseline"}),
                );
                ExecutionStatus::Completed
            }
        }

        let mut executor = CheckpointOnce;
        kernel.process_until_idle(&mut executor);
        let events = kernel.drain_events();
        assert!(
            events
                .iter()
                .any(|event| matches!(event.kind, EventKind::SnapshotIntegrityVerified))
        );

        assert!(!kernel.snapshots.ledger_entries().is_empty());
        kernel
            .snapshots
            .inject_ledger_corruption(0, "deadbeef");

        let follow_up = kernel.allocate_token(1, 0, TokenKind::Plan);
        kernel.submit_token(follow_up);
        kernel.process_until_idle(&mut CheckpointOnce);

        let events = kernel.drain_events();
        assert!(
            events
                .iter()
                .any(|event| matches!(event.kind, EventKind::SnapshotIntegrityViolation))
        );

        let report = kernel
            .last_snapshot_integrity()
            .expect("last integrity report");
        assert!(!report.is_valid);
        assert!(report.failure.is_some());
    }

    struct NoopExecutor;

    impl TokenExecutor for NoopExecutor {
        fn execute(
            &mut self,
            _token: &KernelToken,
            _ctx: &mut ExecutionContext,
        ) -> ExecutionStatus {
            ExecutionStatus::Completed
        }
    }

    #[test]
    fn resource_usage_included_in_started_event() {
        let mut kernel = Kernel::new(9);
        kernel.configure_resource_limit(ResourceType::Cpu, Some(4));

        let token = kernel
            .allocate_token(5, 1, TokenKind::Reason)
            .with_resource_request(crate::resource::ResourceRequest {
                cpu_time: 3,
                memory_bytes: 0,
                network_bytes: 0,
                tokens: 0,
            });

        kernel.submit_token(token);
        kernel.process_until_idle(&mut NoopExecutor);

        let events = kernel.drain_events();
        let started = events
            .iter()
            .find(|event| matches!(event.kind, EventKind::Started))
            .expect("started event recorded");

        let usage = started
            .detail
            .get("resources")
            .and_then(|value| value.get("usage"))
            .expect("resource usage present");

        assert_eq!(usage["cpu"]["consumed"], serde_json::json!(3));
        assert_eq!(usage["cpu"]["limit"], serde_json::json!(4));
    }

    #[test]
    fn resource_violation_emits_event_and_preserves_usage() {
        let mut kernel = Kernel::new(10);
        kernel.configure_resource_limit(ResourceType::Tokens, Some(2));

        let token = kernel
            .allocate_token(5, 1, TokenKind::Plan)
            .with_resource_request(crate::resource::ResourceRequest {
                cpu_time: 0,
                memory_bytes: 0,
                network_bytes: 0,
                tokens: 5,
            });

        kernel.submit_token(token);
        kernel.process_until_idle(&mut NoopExecutor);

        let events = kernel.drain_events();
        assert!(
            events
                .iter()
                .any(|event| matches!(event.kind, EventKind::ResourceViolation))
        );
        assert!(
            !events
                .iter()
                .any(|event| matches!(event.kind, EventKind::Started))
        );

        let usage = kernel.resource_usage();
        assert_eq!(usage.tokens.consumed, 0);
    }
}
