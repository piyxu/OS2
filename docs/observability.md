# Baseline Evaluations & Observability Schema

To keep the PIYXU OS2 0.1.0 version evolution loop safe, the kernel and surrounding tooling must
collect reproducible metrics. This guide defines the minimum viable schema for
metrics, evaluations, and log enrichment so Phase 0 deliverables are satisfied.

## Evaluation Suites

Evaluations run inside forked snapshots and are described by a YAML manifest
containing:

```yaml
goal: "Summarise support conversations"
caps:
  - name: "search"
    tokens: 4000
    invocations: 4
datasets:
  - name: "support-benchmark-v1"
    split: "validation"
metrics:
  - win_rate
  - latency_ms
  - token_cost
```

Each evaluation case records:

- `scenario_id`: deterministic identifier tied to the dataset row.
- `expected_outcome`: human-readable expectation for manual auditing.
- `snapshot_seed`: kernel seed used when creating the sandboxed checkpoint.

These values let the evolver agent compare baseline and candidate performance
without ambiguity.

The Rust implementation backs this specification via the `SnapshotEvaluator`:
- Captures the baseline run plus metrics in a snapshot and forks a candidate
  sandbox from that checkpoint.
- Returns an `EvaluationReport` recording win-rate deltas, latency changes, and
  per-rule `SafetyOutcome`s supplied by configurable `SafetyRule`s.
- Persists both snapshot identifiers so operators can replay either branch when
  auditing a promotion decision.

## Observability Fields

Every `KernelEvent::to_json()` record should be enriched with the following
metadata before it is persisted:

- `run_id`: stable identifier for the evaluation or production run.
- `agent_id`: name or hash of the agent responsible for the token.
- `capability_summary`: optional object describing remaining quotas when the
  event was emitted.
- `sequence`: sequential counter assigned at serialization time to aid
  debugging when multiple events share the same timestamp.
- `chain_hash`: deterministic accumulator digest produced by folding the event
  payload into the snapshot-scoped hash chain.

The kernel already emits detailed event kinds and JSON payloads. New additions
include `scheduled` (token accepted into the priority queue),
`budget_violation` (per-token budget exhausted), explicit `memory_read` /
`memory_write` hooks for semantic memory interactions, `io_queued` /
`io_completed` for deterministic I/O sequencing,
`snapshot_rollback_started` / `snapshot_rollback_committed` /
`snapshot_rollback_failed` for atomic recovery instrumentation,
`snapshot_integrity_verified` / `snapshot_integrity_violation` to capture
automated ledger checks, `resource_violation` when the global governor rejects a
request, `python_vm_sandbox_created` / `python_vm_sandbox_released` to document per-session token budgets, `python_vm_snapshot_tagged` events that stamp each run with the active snapshot identifier, `python_vm_import_*` events whenever the interpreter validates a workspace module hash, `python_vm_env_*` entries that document deterministic environment provisioning, `python_vm_start` / `python_vm_syspath_synced` / `python_vm_complete` whenever the CLI launches
 the embedded interpreter through the `pyvm`, `python`, or `pyx` aliases while recording the invoked command and synchronized search paths in the transcript, emits `python_vm_stream`/`python_vm_session` kernel log events that fold stdout/stderr content into the hash-chained replay timeline with the associated snapshot ID, `python_vm_async_queue_created`/`python_vm_async_queue_drained` entries that document queue lifecycle, `python_vm_async_task_*` events that capture deterministic task execution results, `download_entropy_captured` entries that tie external artifact fetches (for example Hugging Face model downloads) to their
token ledger event identifiers, CAS locations, and entropy bit counts, and `self_task_review_provider_status` / `self_task_review_provider_credentials_updated` / `self_task_review_task_event` records that track provider toggles, credential state, and per-task outcomes for external AI APIs. `started`
events now carry a `resources` object (`requested` and cumulative `usage`),
alongside `capability_handle`, `capability_name`, `tokens_consumed`, and
`tokens_total` so downstream tooling can attribute consumption to both the
global governor and the active capability without consulting the live registry.
Tooling can append the metadata above before writing JSON Lines to disk or
streaming to the observability pipeline. The `kernel_daemon` binary ships a reference
implementation: it loads JSON plans, runs the scripted executor, and persists
`KernelEvent::to_json()` lines suitable for ingestion. The companion
`event_replay` CLI reads these JSON Lines, sorts them deterministically by
timestamp, recomputes the log hash chain to ensure each `chain_hash` matches the
ledger seal, and prints both a timeline and the computed `ExecutionMetrics`
report derived from the log. Historical bare-metal scenarios produced the same event stream by running the `Microkernel` orchestrator in headless mode; Phase 0 archives that path while retaining the identical logging surface inside the Python VM.

## Metrics Aggregation

To satisfy the MVP requirements the following aggregates must be computed per
run:

| Metric        | Definition |
|---------------|------------|
| `win_rate`    | Fraction of evaluation scenarios that produced an acceptable outcome. |
| `latency_ms`  | Milliseconds between the first `started` event and the last `completed`/`yielded` event per scenario. |
| `token_cost`  | Sum of `detail.tokens_consumed` emitted during `started` events. |
| `resource_usage` | Last reported `resources.usage` snapshot from `started` events (captures CPU, memory, network, token consumption vs. limits). |
| `failures`    | Count of `capability_violation`, `resource_violation`, and unexpected error events. |

Aggregates are stored alongside the evaluation manifest and event log in the
snapshot store so regressions are easy to detect and triage. The
`ExecutionMetrics` helper in the Rust crate exposes the same aggregation logic
used by the replay CLI so evaluators and guardrails share a single
implementation.

### RLHF & Routing Telemetry

- `RLHFPipeline` exposes an audit log recording auto-approvals, queued reviews,
  human decisions, and blocking policy names. Persist these entries alongside
  kernel event logs so that governance reviews can trace why a response was
  shipped or halted.
- The deterministic model router surfaces routing outcomes (`endpoint`,
  `reason`, `skipped`) that should be aggregated per run to explain cost/latency
  characteristics and provide evidence when a cloud fallback was required.
- Federated memory synchronization returns `FederatedSyncReport` structures
  enumerating applied, rejected, and stale updates. Logging these reports makes
  replica drift explicit and auditable.

## Replay Checklist

When replaying a run, tooling should validate:

1. All events sort deterministically by `(timestamp, sequence)`.
2. Capability usage counters in the detail payload never exceed budget limits.
3. Metrics recomputed from the replay match the stored aggregates.
4. Log hash chain recomputations match the `chain_hash` field emitted with each event and the seal recorded in the snapshot ledger.
5. Entropy events such as `download_entropy_captured` reference snapshot-linked ledger seals and regenerate the same payload during replay verification, while `python_vm_snapshot_tagged`, `python_vm_syspath_synced`, `python_vm_env_*`, and `python_vm_async_queue_*` events reproduce the interpreter snapshot identifier, search path order, environment provisioning details, and deterministic async queue outcomes.
6. Every `snapshot_rollback_started` event pairs with either a matching
   `snapshot_rollback_committed` or `snapshot_rollback_failed` record whose
   detail references the ledger entry hash restored during the block.
7. Each checkpoint or rollback produces a `snapshot_integrity_verified`
   record; any `snapshot_integrity_violation` is surfaced as a blocker and
   should halt promotion.
8. Any divergence aborts promotion in the self-evolution loop.

These guardrails ensure PIYXU OS2 can evolve safely while providing audit-ready traces
for every decision the agents make.
