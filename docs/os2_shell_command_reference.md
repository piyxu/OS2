# OS2 Shell Command Reference

This reference catalogs every built-in command shipped with the OS2 deterministic shell. Use it alongside the interactive `help` and `man` commands to explore capabilities, auditing requirements, and kernel-specific workflows.

## How to navigate the shell

- Run `help` to list command names with a one-line summary.
- Run `help <command>` or `man <command>` for the usage string and extended description.
- Commands requiring elevated privileges list `admin` or other capability gates below; authenticate with `snapshot-auth` to unlock them.

## Quick reference

| Command | Summary | Usage | Required capabilities | Category |
| --- | --- | --- | --- | --- |
| `audit-diff` | Compare two snapshot ledger events | `audit-diff <left-event-id> <right-event-id> [--context N] [--json]` | process | Ledger & snapshots |
| `bg` | Resume a stopped job in background | `bg <job_id>` | basic | Process management |
| `cap-grant` | Grant a capability | `cap-grant <capability>` | admin | Capability management |
| `cap-list` | List granted capabilities | `cap-list` | basic | Capability management |
| `cd` | Change directory | `cd <path>` | filesystem | Filesystem |
| `config-get` | Read configuration value | `config-get <key>` | basic | Shell configuration |
| `config-set` | Update configuration value | `config-set <key> <value>` | admin | Shell configuration |
| `cp` | Copy file | `cp <source> <destination>` | filesystem | Filesystem |
| `create-env` | Create a deterministic Python environment for sandboxed sessions | `create-env <name> [--description TEXT] [--json] [--no-pip|--with-pip]` | process | Python environments |
| `date` | Show current time | `date` | basic | Environment insight |
| `deterministic-benchmark` | Run the deterministic validation suite and inspect prior runs | `deterministic-benchmark [--json] <status|run> [options]` | admin | Deterministic validation |
| `deterministic-recompile` | Approve code changes via deterministic recompilation | `deterministic-recompile [--json] <queue|pending|approve|history> ...` | admin | Deterministic validation |
| `document-module-tree` | Automate module tree documentation via Roken Assembly | `document-module-tree [--json]` | admin | Documentation tooling |
| `document-release-workflow` | Generate the deterministic release workflow reference | `document-release-workflow [--json]` | admin | Documentation tooling |
| `entropy-audit` | Audit entropy events for deviations | `entropy-audit [--json] [--limit N]` | admin | Observability & security |
| `env` | List environment variables | `env` | basic | Environment insight |
| `fg` | Bring job to foreground | `fg <job_id>` | basic | Process management |
| `git` | Execute Git commands within the workspace | `git <args...>` | filesystem, process | Source control integration |
| `gpu-access` | Acquire or release secure GPU access leases | `gpu-access [--json] <list|acquire|release> ...` | admin | Model & runtime orchestration |
| `hash-ledger` | Inspect or toggle the snapshot ledger read-only guard | `hash-ledger [--json] <status|lock|unlock>` | admin | Ledger & snapshots |
| `help` | List available commands | `help [command]` | basic | Shell basics |
| `integrity-check` | Run integrity hash checks across critical files | `integrity-check [--json] [--label NAME] [paths...]` | admin | Observability & security |
| `jobs` | List background jobs | `jobs` | basic | Process management |
| `kernel-performance` | Report AI kernel energy, memory, and I/O metrics | `kernel-performance [--json] <summary|list|record> ...` | admin | Kernel lifecycle |
| `kernel-ready-flag` | Manage the kernel readiness flag for the next evolution | `kernel-ready-flag [--json] [--set|--clear]` | admin | Kernel lifecycle |
| `kernel-task-rates` | Measure task completion rates from kernel logs | `kernel-task-rates [--json]` | basic | Kernel lifecycle |
| `kernel-updates` | Distribute kernel updates with token-signed packages | `kernel-updates [--json] <list|distribute> ...` | admin | Kernel lifecycle |
| `kill` | Terminate a process | `kill <pid>` | process, admin | Process management |
| `ledger-inspect` | Inspect snapshot ledger events | `ledger-inspect [--limit N] [--kind KIND] [--json]` | process | Ledger & snapshots |
| `living-system` | Transition the kernel into the living deterministic system stage | `living-system [status|transition] [--json] [--refresh] [--notes TEXT] [--operator NAME] [--force]` | admin | Observability & security |
| `load-modules` | Load command modules | `load-modules` | basic | Module management |
| `ls` | List directory contents | `ls [path]` | filesystem | Filesystem |
| `man` | Display detailed command documentation | `man <command>` | basic | Shell basics |
| `mkdir` | Create directory | `mkdir <path>` | filesystem | Filesystem |
| `module-perms` | Manage Python module permissions for capability tokens | `module-perms [--json] <list|grant|revoke> ...` | admin | Module management |
| `module-prune` | Detect and remove unnecessary command modules | `module-prune [--json] [--dry-run]` | admin | Module management |
| `mv` | Move or rename path | `mv <source> <destination>` | filesystem | Filesystem |
| `os2` | OS2 model management and inference | `os2 <model|prompt> ...` | basic | Model & runtime orchestration |
| `os2-dev` | Developer utilities for deterministic model execution | `os2-dev <run|replay|list> [...]` | process | Model & runtime orchestration |
| `pip` | Invoke pip through the embedded Python interpreter | `pip [--resume ID] [--safe] [--token-budget N] [pip-args...]` | process | Python VM & packaging |
| `ps` | List running processes | `ps` | process | Process management |
| `publish-shell-manual` | Publish the deterministic shell technical guide | `publish-shell-manual [--json]` | admin | Documentation tooling |
| `pwd` | Print working directory | `pwd` | filesystem | Filesystem |
| `python` | Route the python alias into the embedded interpreter | `python [--resume ID] [--safe] [--token-budget N] [-c CODE | -m MODULE | script.py [args...]] [--json]` | process | Python VM & packaging |
| `python-verify` | Verify deterministic replay for Python VM sessions | `python-verify [--json] [--limit N]` | admin | Python VM & packaging |
| `pyvm` | Execute code inside the embedded Python VM | `pyvm [--resume ID] [--safe] [--token-budget N] [--eval CODE | --module MOD | script.py [args...]] [--json]` | process | Python VM & packaging |
| `pyx` | Route the pyx alias into the embedded interpreter | `pyx [--resume ID] [--safe] [--token-budget N] [--eval CODE | --module MOD | script.py [args...]] [--json]` | process | Python VM & packaging |
| `rm` | Remove file | `rm <path>` | filesystem | Filesystem |
| `run-script` | Execute batch script | `run-script <path>` | basic | Automation & scripting |
| `secure-backup` | Create signed backups outside the kernel workspace | `secure-backup [--json] create <label> [--token TOKEN] [paths...]` | admin | Observability & security |
| `security-log` | Record and integrate security events | `security-log [--json] <list|record|integrate> ...` | admin | Observability & security |
| `self-feedback` | Analyze recent user interaction transcripts | `self-feedback [--json] <summary|recent> [--limit N]` | admin | Self-evolution workflow |
| `self-task-review` | Manage external AI provider registry and log deterministic task events | `self-task-review [--json] <list|enable|disable|set-key|record> ...` | admin | Self-evolution workflow |
| `set-locale` | Switch shell locale | `set-locale <locale>` | basic | Shell basics |
| `snapshot-auth` | Authenticate admin commands with a snapshot identity | `snapshot-auth <snapshot_id> [--reason TEXT] [--json]` | basic | Ledger & snapshots |
| `snapshot-benchmarks` | Evaluate system behavior with periodic snapshot benchmarks | `snapshot-benchmarks [--json] <status|run> [--force]` | admin | Ledger & snapshots |
| `task-progress` | Display roadmap task completion totals | `task-progress [--json]` | basic | Self-evolution workflow |
| `task-proposals` | Allow Roken Assembly to register roadmap task proposals | `task-proposals [--json] <list|propose> ...` | admin | Self-evolution workflow |
| `time-travel` | Inspect stored snapshot state and diffs | `time-travel [--list | --from ID --to ID] [--json]` | process | Debugging & replay |
| `top` | Show top resource consumers | `top [count]` | process | Process management |
| `touch` | Create empty file | `touch <path>` | filesystem | Filesystem |
| `uname` | Display system information | `uname` | basic | Environment insight |
| `uptime` | Show shell uptime | `uptime` | basic | Environment insight |
| `whoami` | Show current user | `whoami` | basic | Environment insight |

## Automation & scripting

Batch helpers that replay deterministic command sequences from disk without entering the interactive
shell.

### `run-script`

**Summary:** Execute batch script

**Usage:** `run-script <path>`

**Required capabilities:** basic

Reads a newline-delimited script file from the repository root and executes each command
sequentially, preserving deterministic logging for bulk operations.


## Capability management

Commands that expose or inspect capability tokens held by the current session; use them to grant
admin features after policy review.

### `cap-grant`

**Summary:** Grant a capability

**Usage:** `cap-grant <capability>`

**Required capabilities:** admin

Adds a capability string to the in-memory session and records the grant in the audit log. Rate
limited to prevent accidental flooding and typically invoked after snapshot authentication.

### `cap-list`

**Summary:** List granted capabilities

**Usage:** `cap-list`

**Required capabilities:** basic

Echoes the sorted list of capability tokens currently active for the session so you can confirm
whether admin features are unlocked.


## Debugging & replay

Introspection tools for comparing ledger entries or stepping through recorded snapshots without
mutating the live state.

### `time-travel`

**Summary:** Inspect stored snapshot state and diffs

**Usage:** `time-travel [--list | --from ID --to ID] [--json]`

**Required capabilities:** process

Uses `TimeTravelDebugger` to list available snapshots or produce JSON diffs between two states
without mutating the active kernel timeline.


## Deterministic validation

Suites that stress deterministic replay requirements before promoting kernel upgrades; every run
produces signed ledger entries.

### `deterministic-benchmark`

**Summary:** Run the deterministic validation suite and inspect prior runs

**Usage:** `deterministic-benchmark [--json] <status|run> [options]`

**Required capabilities:** admin

Wraps `DeterministicBenchmarkRunner` to execute or inspect the validation suite that replays Python
sessions, stresses AI adapters, and verifies kernel metrics before promotion.

### `deterministic-recompile`

**Summary:** Approve code changes via deterministic recompilation

**Usage:** `deterministic-recompile [--json] <queue|pending|approve|history> ...`

**Required capabilities:** admin

Coordinates the deterministic recompilation queue, allowing operators to list pending work, approve
results, or review history with fully signed ledger entries.


## Documentation tooling

Automated publishers for the living documentation set. Each command regenerates Markdown and records
provenance in the snapshot ledger.

### `document-module-tree`

**Summary:** Automate module tree documentation via Roken Assembly

**Usage:** `document-module-tree [--json]`

**Required capabilities:** admin

Regenerates `docs/module_tree.md` from signed module manifests, verifying signatures and recording
the publication event for provenance.

### `document-release-workflow`

**Summary:** Generate the deterministic release workflow reference

**Usage:** `document-release-workflow [--json]`

**Required capabilities:** admin

Produces the release workflow guide, capturing the generated Markdown and ledger linkage so future
audits can confirm documentation freshness.

### `publish-shell-manual`

**Summary:** Publish the deterministic shell technical guide

**Usage:** `publish-shell-manual [--json]`

**Required capabilities:** admin

Regenerates `docs/piyxu_deterministic_shell.md` and records the publication event so reviewers can
trace documentation updates to specific commands.


## Environment insight

Read-only utilities that display the deterministic shell’s environment so operators can capture
context in audit trails.

### `date`

**Summary:** Show current time

**Usage:** `date`

**Required capabilities:** basic

Prints the deterministic clock timestamp in UTC, mirroring the value written into audit payloads for
reproducibility.

### `env`

**Summary:** List environment variables

**Usage:** `env`

**Required capabilities:** basic

Outputs the shell’s environment variables exactly as injected into Python runs, aiding
reproducibility when replaying sandbox sessions.

### `uname`

**Summary:** Display system information

**Usage:** `uname`

**Required capabilities:** basic

Prints the deterministic OS2 kernel banner so transcripts capture the exact environment signature.

### `uptime`

**Summary:** Show shell uptime

**Usage:** `uptime`

**Required capabilities:** basic

Calculates the duration since the shell session started using the monotonic deterministic clock.

### `whoami`

**Summary:** Show current user

**Usage:** `whoami`

**Required capabilities:** basic

Returns the current shell user (default `os2` or your authenticated username) for audit logs and
scripts.


## Filesystem

Sandboxed filesystem utilities. Every path is constrained to the repository root and returns
deterministic error messages when attempting to escape or operate on invalid objects.

### `cd`

**Summary:** Change directory

**Usage:** `cd <path>`

**Required capabilities:** filesystem

Resolves the target path relative to the workspace root, denies directory escapes, and updates the
shell prompt to the new working directory.

### `cp`

**Summary:** Copy file

**Usage:** `cp <source> <destination>`

**Required capabilities:** filesystem

Copies files within the repository sandbox. Source and destination paths must live under the
workspace; directories are created as needed and binary data is preserved byte-for-byte.

### `ls`

**Summary:** List directory contents

**Usage:** `ls [path]`

**Required capabilities:** filesystem

Lists directory contents within the sandbox, appending `/` to directories and raising deterministic
errors for missing paths or forbidden escapes.

### `mkdir`

**Summary:** Create directory

**Usage:** `mkdir <path>`

**Required capabilities:** filesystem

Creates directories (including parents) beneath the workspace root, blocking attempts to write
outside the sandbox.

### `mv`

**Summary:** Move or rename path

**Usage:** `mv <source> <destination>`

**Required capabilities:** filesystem

Moves or renames files inside the workspace, creating parent directories as needed while preventing
writes outside the sandbox root.

### `pwd`

**Summary:** Print working directory

**Usage:** `pwd`

**Required capabilities:** filesystem

Reports the absolute workspace path for the current directory, ensuring operators know where
filesystem commands operate.

### `rm`

**Summary:** Remove file

**Usage:** `rm <path>`

**Required capabilities:** filesystem

Deletes files inside the sandbox after confirming the target exists and is not a directory. Rate
limited to protect against runaway deletion loops.

### `touch`

**Summary:** Create empty file

**Usage:** `touch <path>`

**Required capabilities:** filesystem

Creates or updates files under the workspace root, ensuring parent directories exist while
preventing path escapes.


## Kernel lifecycle

Rust kernel governance surfaces covering update distribution, readiness signaling, task rate
telemetry, and performance measurements.

### `kernel-performance`

**Summary:** Report AI kernel energy, memory, and I/O metrics

**Usage:** `kernel-performance [--json] <summary|list|record> ...`

**Required capabilities:** admin

Records or inspects kernel energy, memory, and I/O metrics collected by `KernelPerformanceMonitor`.
Supports summary, historical listings, and manual record insertion.

### `kernel-ready-flag`

**Summary:** Manage the kernel readiness flag for the next evolution

**Usage:** `kernel-ready-flag [--json] [--set|--clear]`

**Required capabilities:** admin

Reads or toggles the readiness indicator stored in `cli/data/kernel_state.json`, broadcasting ledger
events when setting or clearing the flag for evolution gating.

### `kernel-task-rates`

**Summary:** Measure task completion rates from kernel logs

**Usage:** `kernel-task-rates [--json]`

**Required capabilities:** basic

Uses `KernelTaskAnalyzer` to compute success and failure rates from kernel log chains, returning
JSON when requested for downstream dashboards.

### `kernel-updates`

**Summary:** Distribute kernel updates with token-signed packages

**Usage:** `kernel-updates [--json] <list|distribute> ...`

**Required capabilities:** admin

Interfaces with `KernelUpdateDistributor` to list available kernel packages or distribute signed
update artifacts with mandatory SHA-256 verification.


## Ledger & snapshots

Ledger readers and snapshot tools. Use these to authenticate sessions, diff events, and execute
recorded benchmark suites.

### `audit-diff`

**Summary:** Compare two snapshot ledger events

**Usage:** `audit-diff <left-event-id> <right-event-id> [--context N] [--json]`

**Required capabilities:** process

Loads `cli/data/snapshot_ledger.jsonl`, computes a structured diff between two event IDs, and prints
changed fields along with surrounding context. Use `--context` to expand the window or `--json` for
machine-readable output.

### `hash-ledger`

**Summary:** Inspect or toggle the snapshot ledger read-only guard

**Usage:** `hash-ledger [--json] <status|lock|unlock>`

**Required capabilities:** admin

Controls the snapshot ledger’s read-only lock. Operators can inspect the current mode, enforce read-
only protection, or unlock for maintenance with full audit trails.

### `ledger-inspect`

**Summary:** Inspect snapshot ledger events

**Usage:** `ledger-inspect [--limit N] [--kind KIND] [--json]`

**Required capabilities:** process

Summarizes ledger statistics, optionally filtering by event kind, and prints the most recent entries
with their indices and timestamps. A JSON mode is available for scripting.

### `snapshot-auth`

**Summary:** Authenticate admin commands with a snapshot identity

**Usage:** `snapshot-auth <snapshot_id> [--reason TEXT] [--json]`

**Required capabilities:** basic

Authenticates the session against a snapshot ID, minting a unique session token, updating
capabilities, and writing a signed ledger event for auditability.

### `snapshot-benchmarks`

**Summary:** Evaluate system behavior with periodic snapshot benchmarks

**Usage:** `snapshot-benchmarks [--json] <status|run> [--force]`

**Required capabilities:** admin

Orchestrates periodic snapshot benchmark runs or reports status, optionally forcing execution when
the scheduler would otherwise defer.


## Model & runtime orchestration

Model registries, runtime planning, and GPU lease commands that prepare deterministic AI workloads
under kernel supervision.

### `gpu-access`

**Summary:** Acquire or release secure GPU access leases

**Usage:** `gpu-access [--json] <list|acquire|release> ...`

**Required capabilities:** admin

Brokers GPU leases through `GPUAccessManager`, letting operators list current allocations, acquire
new ones with expiration metadata, or release stale locks.

### `os2`

**Summary:** OS2 model management and inference

**Usage:** `os2 <model|prompt> ...`

**Required capabilities:** basic

Entrypoint for the model registry and prompt runner. Subcommands install signed model manifests,
list registered capabilities, and execute deterministic prompts via the kernel-bound adapter.

### `os2-dev`

**Summary:** Developer utilities for deterministic model execution

**Usage:** `os2-dev <run|replay|list> [...]`

**Required capabilities:** process

Developer-oriented utilities that replay recorded AI executions, run deterministic prompts against
staging models, and list stored transcripts for debugging.


## Module management

Manage command modules distributed as signed manifests, including permission enforcement and entropy
reduction.

### `load-modules`

**Summary:** Load command modules

**Usage:** `load-modules`

**Required capabilities:** basic

Scans `cli/modules/*.json` for signed module manifests, verifies HMAC signatures, and registers any
exported commands into the shell registry.

### `module-perms`

**Summary:** Manage Python module permissions for capability tokens

**Usage:** `module-perms [--json] <list|grant|revoke> ...`

**Required capabilities:** admin

Manages the module permission registry so capability tokens can be whitelisted against module
imports. Supports JSON output for policy automation.

### `module-prune`

**Summary:** Detect and remove unnecessary command modules

**Usage:** `module-prune [--json] [--dry-run]`

**Required capabilities:** admin

Runs the entropy-based module cleaner to identify unused command modules and optionally remove them
after confirming via the ledger.


## Observability & security

Security instrumentation and audit trails for entropy capture, integrity monitoring, and
backup/signature workflows.

### `entropy-audit`

**Summary:** Audit entropy events for deviations

**Usage:** `entropy-audit [--json] [--limit N]`

**Required capabilities:** admin

Queries the entropy ledger to summarize captured randomness sources, highlight gaps, and optionally
emit structured JSON for downstream policy checks.

### `integrity-check`

**Summary:** Run integrity hash checks across critical files

**Usage:** `integrity-check [--json] [--label NAME] [paths...]`

**Required capabilities:** admin

Runs `IntegrityMonitor` over specified paths (or the default set) to compute hashes, compare them
against stored baselines, and emit deviations to the ledger.

### `living-system`

**Summary:** Transition the kernel into the living deterministic system stage

**Usage:** `living-system [status|transition] [--json] [--refresh] [--notes TEXT] [--operator NAME] [--force]`

**Required capabilities:** admin

Consults `LivingDeterministicSystemManager` to report readiness across kernel telemetry subsystems
or to record the transition into the living-system stage with operator notes.

### `secure-backup`

**Summary:** Create signed backups outside the kernel workspace

**Usage:** `secure-backup [--json] create <label> [--token TOKEN] [paths...]`

**Required capabilities:** admin

Packages selected paths into signed backup archives using `SecureBackupManager`, storing metadata
outside the workspace and recording ledger hooks.

### `security-log`

**Summary:** Record and integrate security events

**Usage:** `security-log [--json] <list|record|integrate> ...`

**Required capabilities:** admin

Lists recorded security events or appends new entries that integrate with the kernel’s security log
hash chain, keeping tamper-evident audit trails.


## Process management

Simulated process table tools that help operators manage foreground/background jobs launched from
the deterministic shell.

### `bg`

**Summary:** Resume a stopped job in background

**Usage:** `bg <job_id>`

**Required capabilities:** basic

Marks a previously launched background job as running again without blocking the shell. Use after
launching commands with `&`; completed jobs return a notice instead of restarting work.

### `fg`

**Summary:** Bring job to foreground

**Usage:** `fg <job_id>`

**Required capabilities:** basic

Joins the thread backing a background job, returning its stdout/stderr and exit status to the
foreground operator once execution completes.

### `jobs`

**Summary:** List background jobs

**Usage:** `jobs`

**Required capabilities:** basic

Summarizes all background jobs spawned via `&`, including job IDs, status, and original command
lines for quick inspection.

### `kill`

**Summary:** Terminate a process

**Usage:** `kill <pid>`

**Required capabilities:** process, admin

Requests that the simulated process table terminate a PID. Requires admin capability to prevent
unreviewed termination attempts.

### `ps`

**Summary:** List running processes

**Usage:** `ps`

**Required capabilities:** process

Prints the simulated process table including PID, state, CPU%, memory, uptime, and friendly names
sourced from `ProcessTable`.

### `top`

**Summary:** Show top resource consumers

**Usage:** `top [count]`

**Required capabilities:** process

Sorts the simulated process table by CPU usage and prints the top entries, optionally limiting the
count to focus on specific workloads.


## Python VM & packaging

The embedded Python interpreter and related tooling. These commands stream output, enforce token
budgets, and capture replay metadata for every run.

### `pip`

**Summary:** Invoke pip through the embedded Python interpreter

**Usage:** `pip [--resume ID] [--safe] [--token-budget N] [pip-args...]`

**Required capabilities:** process

Routes arguments through `python -m pip`, inheriting token-budget controls, safe-mode toggles, and
interactive streaming so package installs remain auditable.

### `python`

**Summary:** Route the python alias into the embedded interpreter

**Usage:** `python [--resume ID] [--safe] [--token-budget N] [-c CODE | -m MODULE | script.py [args...]] [--json]`

**Required capabilities:** process

Executes the embedded Python interpreter with support for `-c`, `-m`, scripts, safe mode, resume-
from-snapshot, token budget overrides, JSON summaries, and interactive streaming tied to ledger
events.

### `python-verify`

**Summary:** Verify deterministic replay for Python VM sessions

**Usage:** `python-verify [--json] [--limit N]`

**Required capabilities:** admin

Runs the deterministic replay verifier that hashes stdout/stderr across prior Python sessions, flags
divergences, and emits aggregate verification metrics.

### `pyvm`

**Summary:** Execute code inside the embedded Python VM

**Usage:** `pyvm [--resume ID] [--safe] [--token-budget N] [--eval CODE | --module MOD | script.py [args...]] [--json]`

**Required capabilities:** process

Alias optimized for explicit evaluator flags. Mirrors `python` but omits `-c`, making
`--eval`/`--module` semantics explicit for sandbox automation.

### `pyx`

**Summary:** Route the pyx alias into the embedded interpreter

**Usage:** `pyx [--resume ID] [--safe] [--token-budget N] [--eval CODE | --module MOD | script.py [args...]] [--json]`

**Required capabilities:** process

Convenience alias that mirrors `pyvm` while accepting `-c` to match historical tooling. Useful when
porting scripts that expect the legacy interface.


## Python environments

Provision reproducible Python environments for sandboxed execution, ensuring metadata is captured in
the ledger.

### `create-env`

**Summary:** Create a deterministic Python environment for sandboxed sessions

**Usage:** `create-env <name> [--description TEXT] [--json] [--no-pip|--with-pip]`

**Required capabilities:** process

Invokes `PythonEnvironmentManager` to materialize a new sandbox in `cli/python_vm/sandboxes/`,
optionally skipping pip bootstrapping. Emits ledger event IDs, execution timing, and metadata paths.


## Self-evolution workflow

Governance rails that collect feedback, review task proposals, and summarize roadmap progress for
self-developing upgrades.

### `self-feedback`

**Summary:** Analyze recent user interaction transcripts

**Usage:** `self-feedback [--json] <summary|recent> [--limit N]`

**Required capabilities:** admin

Runs the feedback analyzer over recent transcripts to surface operator sentiment, completion stats,
and actionable insights for the self-evolution loop.

### `self-task-review`

**Summary:** Manage external AI provider registry and log deterministic task events

**Usage:** `self-task-review [--json] <list|enable|disable|set-key|record> ...`

**Required capabilities:** admin

Manages the self-task review module by enabling providers, rotating API keys, or recording
deterministic task completions with ledger references.

### `task-progress`

**Summary:** Display roadmap task completion totals

**Usage:** `task-progress [--json]`

**Required capabilities:** basic

Summarizes roadmap completion percentages derived from the signed task list so stakeholders can
gauge milestone status.

### `task-proposals`

**Summary:** Allow Roken Assembly to register roadmap task proposals

**Usage:** `task-proposals [--json] <list|propose> ...`

**Required capabilities:** admin

Interfaces with the task proposal registry, listing stored suggestions or registering new proposals
complete with description and tags.


## Shell basics

Orientation commands that mirror standard UNIX ergonomics while providing localized summaries and
usage hints.

### `help`

**Summary:** List available commands

**Usage:** `help [command]`

**Required capabilities:** basic

Lists commands with localized summaries. When invoked with a name, it prints the resolved usage
string so you can see required arguments without leaving the shell.

### `man`

**Summary:** Display detailed command documentation

**Usage:** `man <command>`

**Required capabilities:** basic

Formats a manual-style page for the requested command, including synopsis, description, and
capability requirements drawn from the registry.

### `set-locale`

**Summary:** Switch shell locale

**Usage:** `set-locale <locale>`

**Required capabilities:** basic

Switches the active locale so summaries, usage strings, and documentation reflect translated content
when available.


## Shell configuration

Per-session configuration knobs for adjusting interpreter budgets or other deterministic shell
defaults.

### `config-get`

**Summary:** Read configuration value

**Usage:** `config-get <key>`

**Required capabilities:** basic

Reads the deterministic shell configuration dictionary (for example `script_token_budget`) and
prints the stored value or an error if the key is missing.

### `config-set`

**Summary:** Update configuration value

**Usage:** `config-set <key> <value>`

**Required capabilities:** admin

Updates configuration entries inside the live session—most notably the script token budget override.
Requires admin rights to keep policy changes auditable.


## Source control integration

Safe wrappers over Git that preserve streaming output and interrupt handling inside the sandbox.

### `git`

**Summary:** Execute Git commands within the workspace

**Usage:** `git <args...>`

**Required capabilities:** filesystem, process

Streams Git subprocess output directly to the terminal while preserving audit transcripts. Supports
`clone`, `status`, and other safe operations confined to the workspace.


