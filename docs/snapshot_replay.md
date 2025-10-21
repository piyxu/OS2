# Snapshot Storage & Replay Specification

The snapshot engine (`rust/os2-kernel/src/snapshot.rs`) offers a deterministic
persistence layer that captures reversible checkpoints of agent state. This
specification documents the data model and lifecycle for checkpoints so the
kernel, tooling, and host runtimes follow the same conventions.

## Data Model

Each checkpoint is represented by a `SnapshotState` struct with the fields
below:

| Field      | Type                | Description |
|------------|---------------------|-------------|
| `id`       | `SnapshotId`        | Monotonic identifier assigned at creation time. |
| `label`    | `String`            | Human-readable tag supplied by the executor. |
| `state`    | `serde_json::Value` | Arbitrary state payload serialised as JSON. |
| `timestamp`| `u64`               | Deterministic virtual timestamp captured from the kernel clock. |

Snapshots are stored inside an append-only `BTreeMap` keyed by `SnapshotId`.
The most recent snapshot is fast to retrieve via `SnapshotEngine::latest()`.

## Lifecycle Operations

1. **Checkpoint**: Executed by calling `ExecutionContext::checkpoint`. The kernel
   assigns a fresh identifier, records the JSON payload, and emits a
   `snapshot_created` event that links the originating token to the snapshot.
2. **Restore**: Read-only access through `Kernel::restore(id)` which returns the
   captured `SnapshotState`. Restores are side-effect free—callers must decide
   how to apply the data.
3. **Iteration**: Tooling can inspect all snapshots using
   `SnapshotEngine::iter()` (exposed via `SnapshotEngine::states()` in the crate)
   to build dashboards or replay timelines.

Because snapshots are content-addressed by identifier and label, the kernel does
not mutate historical records. Rollbacks are executed by restoring the JSON
payload into the appropriate agent runtime and replaying events from the
associated timestamp forward.

## Storage Layout

In the MVP the engine stores snapshots in memory. For persistence, tooling
should serialise the following JSON object per checkpoint:

```json
{
  "snapshot_id": 3,
  "label": "reflect-after-action",
  "timestamp": 124,
  "state": {"memory": "vector://xyz"}
}
```

Persisted checkpoints may live alongside the JSON event stream described in
`kernel_protocol.md`. The pair of files provides sufficient material to fully
replay any execution: snapshots recreate state while events reproduce
interactions and capability decisions.

## Replay Workflow

1. Load stored snapshots into a map keyed by `snapshot_id`.
2. Feed recorded events (sorted by `timestamp`) back into a dry-run kernel.
3. When encountering a `snapshot_created` event, verify the snapshot payload
   matches the restored state to guarantee integrity.
4. Reconstruct agent decisions using the sequence of `custom` events and
   capability usage metadata from the event details.

This reproducible workflow ensures PIYXU OS2 can offer audit-ready explanations for
any agent behaviour and supports rollback-on-failure semantics during evolution
trials.
