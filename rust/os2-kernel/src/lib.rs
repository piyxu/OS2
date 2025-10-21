//! PIYXU OS2 0.1.0 version deterministic microkernel core implemented in Rust.

pub mod capability;
pub mod capability_chain;
pub mod clock;
pub mod entropy;
pub mod event_bus;
pub mod evolution;
pub mod io_queue;
pub mod kernel;
pub mod memory;
pub mod metrics;
pub mod microkernel;
pub mod model_routing;
pub mod python_host;
pub mod resource;
pub mod resource_monitor;
pub mod rlhf;
pub mod scheduler;
pub mod security;
pub mod snapshot;
pub mod symbolic;
pub mod telemetry;
pub mod token_signing;
pub mod tooling;
pub mod wasm_host;

pub use capability::{
    CapabilityError, CapabilityHandle, CapabilityInfo, CapabilityLimits, CapabilityRegistry,
    CapabilityUsage,
};
pub use capability_chain::{CapabilityGrantLedger, CapabilityGrantRecord};
pub use clock::DeterministicClock;
pub use entropy::EntropyBalancer;
pub use event_bus::{EventBuilder, EventBus, EventKind, KernelEvent};
pub use evolution::{
    EvaluationReport, EvaluationRunResult, EvolverAgent, EvolverProposal, LearningSample,
    LearningSummary, SafetyOutcome, SafetyRule, SelfEvolutionReport, SelfEvolutionTrigger,
    SnapshotEvaluator,
};
pub use io_queue::{CompletedIoOperation, IoOperationKind, IoOperationRecord, IoQueue};
pub use kernel::{
    AsyncBlock, AsyncBlockReport, AsyncSchedule, AsyncScheduledToken, AsyncStep, AsyncStepReport,
    ExecutionContext, ExecutionStatus, Kernel, SnapshotOperationError, TokenExecutor,
};
pub use memory::{
    EpisodicRecord, FederatedChange, FederatedSyncReport, FederationError, LongTermRecord,
    SemanticMemoryBus, ShortTermRecord,
};
pub use metrics::{CapabilityUsageMetrics, ExecutionMetrics, TokenKindMetrics, TokenRunMetrics};
pub use microkernel::{Microkernel, MicrokernelConfig, MicrokernelImage, MicrokernelService};
pub use model_routing::{
    DeterministicRouter, EndpointKind, ModelEndpoint, RouteAssignment, RouteDecision, RouteReason,
    RoutingRequest,
};
pub use python_host::{PipelineRun, PipelineSpec, PythonHost};
pub use resource::{
    ResourceError, ResourceGovernor, ResourceQuota, ResourceRequest, ResourceType,
    ResourceUsageSnapshot,
};
pub use resource_monitor::{ResourceObservation, ResourceObserver};
pub use rlhf::{
    AuditEntry, AuditOutcome, InteractionInput, PendingHumanReview, PipelineDecision, PolicyAction,
    PolicyRule, RLHFPipeline, ReviewDecision, ReviewId,
};
pub use scheduler::{KernelToken, Scheduler, SchedulerError, TokenBudget, TokenId, TokenKind};
pub use security::{
    ModuleMetadata, ModuleSecurityError, ModuleSecurityManager, ModuleSigningPolicy,
};
pub use snapshot::{
    SnapshotDiffEntry, SnapshotDiffLedger, SnapshotEngine, SnapshotId, SnapshotLedger,
    SnapshotLedgerEntry, SnapshotLedgerIntegrityFailure, SnapshotLedgerIntegrityFailureKind,
    SnapshotLedgerIntegrityReport, SnapshotRollbackError, SnapshotRollbackRecord, SnapshotState,
};
pub use symbolic::{
    SymbolicAtom, SymbolicDerivation, SymbolicError, SymbolicLogicEngine, SymbolicRule,
    SymbolicTerm,
};
pub use telemetry::{TelemetryFrame, TelemetrySynchronizer};
pub use token_signing::{
    TokenSignature, TokenSignatureLedger, TokenSignatureRecord, TokenSigningError,
};
pub use tooling::{ToolAdapter, ToolCatalog, ToolInvocationResult};
pub use wasm_host::{WasmSidecar, WasmSidecarConfig};
