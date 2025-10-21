# Kernel Messaging & Capability Protocol

This document grounds the PIYXU OS2 0.1.0 version kernel messaging contract in the concrete
structures that ship in the `os2-kernel` crate. The goal is to make
interactions deterministic and replayable across host runtimes.

## Token Envelope

Reasoning work is scheduled as `KernelToken` records. Each token contains the
fields below (see `rust/os2-kernel/src/scheduler.rs`):

| Field        | Type        | Description |
|--------------|-------------|-------------|
| `id`         | `TokenId`   | Monotonic identifier allocated by the kernel. |
| `priority`   | `u8`        | Higher values execute first, with FIFO ordering inside the same priority. |
| `cost`       | `u64`       | Estimated capability budget consumed when the token runs. |
| `kind`       | `TokenKind` | Enumerates `Perception`, `Reason`, `Plan`, `Act`, `Reflect`. |
| `capability` | `Option<CapabilityHandle>` | Optional capability required to execute. |
| `granted_capabilities` | `Vec<CapabilityHandle>` | Metadata describing capabilities visible to the executor. |
| `context_hash` | `String` | Stable hash of the token's working context. |
| `goal`       | `Option<String>` | Human-readable intent associated with the token. |
| `dependencies` | `Vec<TokenId>` | Token IDs that must complete before this token can run. |
| `budget`     | `TokenBudget` | `{ limit, consumed }` structure enforcing per-token cost ceilings. |
| `resources`  | `ResourceRequest` | `{ cpu_time, memory_bytes, network_bytes, tokens }` request enforced by the global resource governor. |
| `payload`    | `serde_json::Value` | Arbitrary JSON payload forwarded to the executor. |

Tokens submitted through `Kernel::submit_token` are persisted in a deterministic
priority queue. Schedulers are replay-safe because insertion order is captured as
an explicit sequence counter, guaranteeing the same execution order across runs
with identical inputs.

## Event Stream Schema

Every lifecycle transition results in a `KernelEvent`. The kernel exposes a
`KernelEvent::to_json()` helper that returns an audit-friendly JSON object:

```json
{
  "token_id": 7,
  "timestamp": 33,
  "kind": "custom",
  "label": "execution",
  "detail": {"ok": true}
}
```

Events carry a monotonic virtual timestamp obtained from the deterministic
clock. The `kind` field is normalized to one of the following values:

| Kind                 | Meaning |
|----------------------|---------|
| `scheduled`          | Token was accepted into the queue with its metadata. |
| `started`            | Token execution began after capability checks, resource governor accounting, and budget enforcement. Includes capability attribution fields. |
| `completed`          | Executor returned `ExecutionStatus::Completed`. |
| `yielded`            | Executor yielded another token back to the scheduler. |
| `capability_violation` | Capability budgets prevented the token from running. |
| `budget_violation`   | Scheduler rejected the token because its per-token budget was exhausted. |
| `resource_violation` | Global resource governor rejected the request (CPU, memory, network, or tokens). |
| `snapshot_created`   | `ExecutionContext::checkpoint` produced a new snapshot. |
| `memory_write`       | Executor wrote to the semantic memory bus. |
| `memory_read`        | Executor read from the semantic memory bus. |
| `custom`             | Executor-emitted events carrying the custom `label`. |

Events are safe to persist as JSON Lines. Replaying a run involves loading the
stored sequence, sorting by `timestamp`, and refeeding the associated tokens
into the kernel. Because the timestamps and seeds are deterministic, this
replay will mirror the original execution.

### `started` Event Detail

When a token begins executing the kernel enriches the `detail` payload with the
resource request/usage snapshot and any capability attribution:

```json
{
  "resources": {
    "requested": {
      "cpu_time": 3,
      "memory_bytes": 0,
      "network_bytes": 0,
      "tokens": 2
    },
    "usage": {
      "cpu": {"limit": 12, "consumed": 3},
      "memory": {"limit": null, "consumed": 0},
      "network": {"limit": null, "consumed": 0},
      "tokens": {"limit": 20, "consumed": 2}
    }
  },
  "capability_handle": 1,
  "capability_name": "search",
  "invocations": 3,
  "tokens_total": 1200,
  "tokens_consumed": 400
}
```

`resources.requested` mirrors the `ResourceRequest` on the token and
`resources.usage` captures the cumulative governor snapshot after accepting the
request. `tokens_total` reflects the cumulative capability usage after this
invocation, while `tokens_consumed` shows the delta for the current token. These
fields let the metrics collector compute per-capability spend and global
resource consumption without consulting the live registry.

### `resource_violation` Event Detail

If the global governor rejects a request the kernel emits:

```json
{
  "error": "resource_budget_exceeded",
  "resource": "tokens",
  "attempted": 5,
  "remaining": 2
}
```

The `resource` field enumerates `cpu`, `memory`, `network`, or `tokens`. Because
the token never starts, no resource usage is consumed and the event stream
contains only `scheduled` → `resource_violation` entries for the rejected token.

## Capability Format

Capabilities are registered with explicit budgets inside the
`CapabilityRegistry` (`rust/os2-kernel/src/capability.rs`). Each entry tracks:

- `handle`: stable identifier referenced by tokens.
- `limits.max_invocations`: optional ceiling on invocations per run.
- `limits.max_tokens`: optional ceiling on aggregate token usage per run.
- `usage`: counter structure storing consumed invocations and tokens.

The registry enforces budgets before a token executes. When a violation occurs
`Kernel::process_next` emits a `capability_violation` event containing a
machine-readable error payload (`not_found`, `invocation_budget_exceeded`, or
`token_budget_exceeded`).

### Example Capability Declaration

```rust
let search_cap = kernel.register_capability(
    "search",
    CapabilityLimits {
        max_invocations: Some(100),
        max_tokens: Some(20_000),
    },
);
```

Tokens that require this capability call `.with_capability(search_cap)` to bind
the handle. Usage counters are visible to executors via
`ExecutionContext::capability_usage`, enabling adaptive scheduling strategies.

## Execution Timeline Example

A typical replay log for a single token looks like:

1. `scheduled`: token metadata was registered with the scheduler.
2. `started`: capability and per-token budgets passed, executor invoked.
3. `memory_write` / `memory_read`: executor interacted with semantic memory (optional).
4. `custom`/`label=execution`: executor emitted domain-specific metadata.
5. `snapshot_created`: executor called `checkpoint` and recorded state (optional).
6. `completed`: token finished and was removed from the scheduler.

Capturing this sequence alongside the kernel seed and the serialized
`snapshot` payload enables full deterministic replays and audit trails.

## Federated Memory Replication

Replicas synchronize long-term semantic memory using CRDT-style `FederatedChange`
records:

| Field | Type | Description |
|-------|------|-------------|
| `origin` | `String` | Replica identifier that produced the change. |
| `key` | `String` | Long-term record identifier. |
| `value` | `serde_json::Value` | Updated payload. |
| `version` | `u64` | Lamport-style clock to enforce deterministic ordering. |
| `capability` | `String` | Capability required to apply the change. |

`SemanticMemoryBus::apply_federated_changes` returns a `FederatedSyncReport`
summarizing merges:

| Field | Description |
|-------|-------------|
| `applied` | Count of updates merged into the local replica. |
| `stale` | Keys rejected because a newer version already exists. |
| `rejected` | `{ key, reason }` tuples describing capability or policy violations. |

Replicas must be registered with allowed capabilities before changes are
accepted. Unauthorized or stale updates are surfaced via the report so guardrail
agents can alert operators.

## Deterministic Model Routing

High-level runtimes select inference endpoints through the deterministic router.
Each `RouteDecision` captures the selected endpoint and the reason:

| Field | Type | Description |
|-------|------|-------------|
| `endpoint` | `ModelEndpoint` | Selected local/cloud deployment including latency/cost weights and capability list. |
| `reason` | `RouteReason` | `Primary` when preferred endpoint had budget, or `BudgetFallback` with the skipped endpoints. |

Rejected routes include a diagnostic string enumerating exhausted candidates.
Persisting these decisions alongside kernel logs makes it clear why a run fell
back to a cloud provider or failed outright.

## RLHF Audit Records

`RLHFPipeline` maintains an ordered audit log with `AuditEntry` items:

| Field | Type | Description |
|-------|------|-------------|
| `interaction_id` | `String` | Identifier tying the decision back to kernel events. |
| `outcome` | `AuditOutcome` | One of `AutoApproved`, `Blocked { policy }`, `QueuedForReview { review_id, reviewer }`, `HumanApproved`, or `HumanRejected`. |
| `policies` | `Vec<String>` | Policy names that matched the interaction. |

These records act as governance artifacts for the evolution loop and are stored
beside event logs and evaluation reports.
