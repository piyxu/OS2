# OS2 Technical Overview

OS2 is a deterministic playground that lets AI agents run inside a capability-governed microkernel without jeopardising the host machine. This reference curates the previously scattered documentation into a single, production-ready brief so newcomers can see how the shipped system fits together and where to dive deeper.

---

## 1. Architecture Snapshot

### 1.1 Kernel (Rust `rust/os2-kernel/`)
- **Deterministic scheduler.** Consumes `KernelToken` submissions, honours priority, token, memory, network, and wall-clock budgets, and advances jobs through a monotonic virtual clock.
- **Capability registry.** Tracks capability descriptors, grants, revocations, and scoped configuration such as rate limits and token ceilings.
- **Event bus & ledger.** Emits `EventEnvelope` records into a hash-chained ledger, guaranteeing tamper-evident auditing for every capability invocation and state transition.
- **Snapshot engine.** Captures copy-on-write checkpoints, supports labelled snapshots, and exposes replay handles for deterministic rollbacks.

### 1.2 Python Host (`cli/`)
- **Embedded CPython runtime.** Launches sandboxed interpreters (`python`, `pyvm`, `pyx`) that stream stdout/stderr deterministically and respect per-run token budgets supplied by the kernel.
- **Deterministic shell.** Provides a readline-powered REPL, background jobs, pipelines, and capability-aware wrappers for `git`, `pip`, `curl`, and custom automation.
- **Module sidecars.** Optional WASM modules and Python packages can be mounted through audited capability grants, letting agents extend the kernel surface safely.

### 1.3 Persistence & Federation
- **Snapshot store.** Maintains JSON-serialised `SnapshotState { id, label, state, timestamp }` payloads for deterministic recovery.
- **Semantic memory buses.** Support vector, episodic, and key/value memories with versioned commits.
- **Federated replication.** Exchanges CRDT-style `FederatedChange` bundles, refusing stale or unauthorised updates based on capability policy.

---

## 2. Repository & Module Map
- `cli/` – deterministic shell core, Python VM launchers, regression harnesses, and CLI transcripts under `cli/logs/`.
- `rust/os2-kernel/` – microkernel scheduler, capability registry, event bus, snapshot engine, and ledger implementation.
- `docs/` – this overview, the curated journey log (`project_journey.md`), capability limits (`CapabilityLimits.toml`), and localisation pointer (`yol_hikayesi.md`).
- `scripts/` – automation helpers such as `document-module-tree`, `kernel-ready-flag`, and deterministic benchmark wrappers.
- `releases/` – manifests, reproducible build artefacts, and snapshot bundles promoted by the release workflow.

---

## 3. Capability & Scheduler Management
- **Registration.** Capabilities declare identifiers, invocation shapes, token ceilings, and optional rate limits. Grants are versioned so revocation immediately blocks new scheduling attempts.
- **Token lifecycle.** `Kernel::submit_token` accepts work units tagged with capability requirements, dependency edges, and budget envelopes. Tokens advance through `scheduled → started → (custom events) → snapshot_created → completed`.
- **Violation handling.** Budget overruns (`resource_budget_exceeded`) and policy breaches (`capability_violation`) halt execution, emit structured ledger entries, and preserve the last consistent snapshot.
- **Introspection.** Shell helpers (`cap-list`, `cap-show <id>`) surface live grant state, usage counters, and pending revocations for operators.

---

## 4. Kernel Protocol Summary
- **Requests.** Primary verbs include `SubmitToken`, `CancelToken`, `CreateSnapshot`, `ListSnapshots`, `GrantCapability`, `RevokeCapability`, and `StreamEvents`.
- **Responses.** Each request yields a `KernelResponse` with `accepted`, `rejected`, or `queued` dispositions plus ledger hashes for audit correlation.
- **Event envelopes.** Every state transition produces an `EventEnvelope { event_id, hash, prev_hash, capability_id, payload }`. Deterministic seeds and timestamps allow byte-for-byte replay.
- **Streaming.** Clients subscribe via `StreamEvents` to receive structured JSON records. Divergences in the hash chain raise integrity alerts before promotions are finalised.
- **Snapshot replay.** `ReplaySnapshot` streams deterministic seeds, capability invocations, and shell transcripts to reconstruct historical runs exactly.

---

## 5. Snapshot & Replay Operations
1. **Create.** Operators invoke `snapshot-create --label <tag>` or schedule automatic checkpoints from capability policies.
2. **Authenticate.** `snapshot-auth <id>` mints signed tokens binding ledger hashes to snapshot content for distribution.
3. **Inspect.** `snapshot-show <id>` dumps metadata, while `snapshot-diff <a> <b>` compares deterministic state deltas.
4. **Replay.** `snapshot-replay <id>` restores state into a temporary sandbox, re-emitting event streams for verification.
5. **Promote.** Verified snapshots are marked ready via `kernel-ready-flag --set <id>` and archived under `releases/`.

---

## 6. Shell Command Reference (Essentials)
| Command | Purpose | Notes |
| --- | --- | --- |
| `python [--token-budget N] script.py` | Run scripts inside the audited CPython VM. | Streams output, honours ledger logging, supports snapshot tagging via `--tag`.
| `pyvm run <module>` | Launch named Python module entrypoints. | Uses deterministic seeds injected by the scheduler.
| `pyx <expr>` | Evaluate quick Python expressions. | Writes results and token usage to session transcripts.
| `pip install <pkg>` | Capability-checked package installation. | Mirrors wheels, records hashes, and respects outbound network policies.
| `git <subcommand>` | Safe git wrapper with deterministic progress. | Enforces allow-list remotes and logs all refs fetched.
| `run-script <path>` | Execute batch command files with conditional logic. | Stops on first failure unless `--keep-going` is supplied.
| `cap-list` / `cap-show <id>` | Inspect active capability grants. | Surfaces usage counters, budgets, and revocation flags.
| `ledger-inspect [--since HASH]` | Stream ledger events for auditing. | Supports JSON or table output.
| `snapshot-*` family | Manage deterministic checkpoints. | `snapshot-create`, `snapshot-auth`, `snapshot-replay`, `snapshot-diff`.
| `deterministic-benchmark run` | Execute reproducible workloads. | Emits JSON reports stored under `releases/<tag>/benchmarks/`.

Transcripts for every command run are persisted under `cli/logs/<timestamp>.jsonl`, enabling deterministic replays and audits.

---

## 7. Observability & Safety Nets
- **Structured logging.** Kernel, shell, and Python VM emit JSON records capturing command, seed, capability, budgets, and outcomes.
- **Metrics.** Prometheus-style counters track token throughput, snapshot duration, violation counts, and capability latency percentiles.
- **Alerts.** Ledger integrity failures, capability violations, and policy breaches emit `alert` events consumed by operators.
- **Supply chain.** Downloaded artefacts bind to ledger hashes. Verification commands (`ledger-verify`, `snapshot-auth`) confirm provenance before promotion.
- **Deterministic routing.** Model selection and external API usage log the scoring rationale so policy reviews can replay decisions.

---

## 8. Release Workflow
1. **Prepare.** Run `deterministic-benchmark run --json` and `document-module-tree --json` to capture performance and module inventories.
2. **Audit.** Review ledger diffs with `ledger-inspect --since <prev_release_hash>` and confirm snapshot integrity via `snapshot-verify`.
3. **Document.** Publish updated docs and CLI manuals directly from the shell to keep operator guidance in sync with the release.
4. **Flag.** Execute `kernel-ready-flag --set <snapshot_id>` once validation, documentation, and benchmarks have passed.
5. **Archive.** Store manifests, snapshot bundles, and benchmark artefacts under `releases/<version>/` for reproducible recovery.

---

## 9. Evaluation & Continuous Improvement
- **Scenario suites.** `deterministic-benchmark` and `scenario-run` replay ledger-backed workloads to measure latency, determinism, and policy compliance.
- **Safety gating.** Proposed self-evolution loops run inside snapshots; only successful evaluations promote to the canonical ledger.
- **Community feedback.** Operators log findings in the project journey and open capability requests through the audited shell commands.

---

## 10. Additional References
- Capability limits – [`docs/CapabilityLimits.toml`](CapabilityLimits.toml)
- Milestone history – [`docs/project_journey.md`](project_journey.md)
- Localised pointer – [`docs/yol_hikayesi.md`](yol_hikayesi.md)
- Rust microkernel – [`rust/os2-kernel/`](../rust/os2-kernel/)
- Deterministic shell – [`cli/`](../cli/)

