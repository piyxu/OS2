# PIYXU OS2 0.1.0 version Kernel Management and Command Quick Reference

## Shell Commands
- **help / man**: list commands and view manual entries.
- **set-locale**: switch interface language.
- **pwd / ls / cd / mkdir / touch**: navigate and prepare directories within the sandbox.
- **rm / mv / cp**: remove, move, or copy repository files while staying under the root.
- **ps / top / kill**: inspect or terminate simulated kernel processes.
- **jobs / fg / bg**: monitor background jobs and resume them in the foreground or background.
- **uname / whoami / env / date / uptime**: capture environment metadata for audit trails.
- **cap-list / cap-grant**: review or extend capability sets (grants require admin capability).
- **config-get / config-set**: inspect or adjust shell configuration values.
- **run-script**: execute batch scripts with conditional blocks (respects token budget).
- **pyvm**: launch the embedded Python VM, capturing ledger-backed start/syspath/complete events, recording stdout/stderr in the kernel log with hash-chained `python_vm_stream`/`python_vm_session` entries, and returning deterministic summaries.
- **python / pyx**: route familiar aliases into the embedded interpreter while logging the invoked command and synchronized search paths for replay.
- **load-modules**: register JSON-defined command modules from `cli/modules`.
- **module-prune**: scan command modules for entropy-reducing issues and delete unnecessary definitions.
- **living-system**: review readiness across kernel telemetry subsystems and record the transition into the living deterministic system stage with a ledger event.

## Kernel Development Workflow
1. Compile the Rust kernel daemon through `make build-kernel`; the wrapper targets a Linux-style cargo triple by default and writes artifacts to `target/$(CARGO_TARGET)/release/` outside of source control.
2. Execute the Python smoke checks with `make run-python-host`, which injects `KERNEL_BUILD_DIR` and `PYTHONPATH=$(pwd)` so the command shell only runs after a fresh local build.
3. Treat the Python shell (`cli.command_shell`) as the canonical kernel VM. Iterate on capabilities and deterministic execution directly within this environment.
4. Build Python packaging artifacts with `python -m build` and verify outputs remain text-based.
5. Kernel sources live in `rust/os2-kernel`; run `cargo check` or `cargo test` from that directory when iterating on Rust code.
6. Structured CLI transcripts are written to `rust/os2-kernel/logs/cli_sessions/`; inspect them when auditing kernel interactions.
7. Keep repository contributions text-only—avoid adding binary artifacts anywhere in source control. Use `make clean` (or `cargo clean`) to remove local build products and `__pycache__` directories before committing.

## Maintenance Tips
- Record new operational commands in `docs/command_interface.md` when expanding shell capabilities; mirror them in this quick reference.
- Document kernel evolution milestones in `docs/kernel_protocol.md` and related design notes under `docs/`.
- Seal every entropy capture by recording the snapshot digest alongside the Roken Assembly beacon in the ledger notes; replay tools now verify this link automatically.
- Confirm that external model installs emit `download_entropy_captured` ledger entries tying artifact hashes, CAS paths, and token ledger IDs together; investigate any missing entropy records before promoting changes.
- Ensure each Python VM run writes a `python_vm_snapshot_tagged` ledger event and that transcripts/kernel logs echo the associated snapshot ID before approving CLI-driven changes.
- Recompute the snapshot-scoped log hash chain during audits and confirm the ledger seal matches `chain_hash` values emitted in observability logs.
- Keep `cli/python_vm/import_manifest.json` synchronized with trusted workspace modules; update hashes after modifying Python files that sandboxed sessions should import so verification events remain deterministic.
- Treat any `snapshot_integrity_violation` event as a release blocker; rerun the audit, investigate tampering, and only resume once a `snapshot_integrity_verified` record confirms the ledger chain.

## Bootable Image Validation (Archived)
- Bare-metal validation steps have been removed from the active workflow. Historical experiments are no longer documented; begin and end kernel work with `make run-python-host` inside the deterministic Python environment.
