"""Documentation helpers for Phase 7 release tasks."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from textwrap import dedent
from typing import Dict, List

from .snapshot_ledger import SnapshotLedger, SnapshotLedgerError


def _isoformat_utc(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


class DocumentationError(RuntimeError):
    """Raised when documentation artifacts cannot be published."""


def _write_text(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")


@dataclass
class DocumentationPublisher:
    """Publishes documentation artifacts for the deterministic shell release."""

    root: Path
    snapshot_ledger: SnapshotLedger

    def publish_shell_manual(self) -> Dict[str, object]:
        """Write the deterministic shell manual to the docs directory."""

        target = self.root / "docs" / "piyxu_deterministic_shell.md"
        body = dedent(
            """
            # PIYXU Deterministic Shell Technical Guide

            The PIYXU deterministic shell is the curated command surface for the
            Codex-driven AI kernel. The environment exposes a reproducible CLI that
            binds every command invocation to the snapshot ledger, kernel log, and
            replay subsystems so the task list can be executed deterministically.

            ## Architectural Overview

            ```mermaid
            graph TD
              A[Operator Command] --> B[Deterministic Shell]
              B --> C[Snapshot Ledger]
              B --> D[Kernel Log]
              B --> E[Python VM Sandbox]
              C --> F[Replay Verifier]
              D --> F
              E --> F
            ```

            ## Primary Subsystems

            - **Snapshot Ledger Integration** — every command records a signed event
              so replay tooling can reconstruct state transitions exactly.
            - **Python VM Sandbox** — commands such as `pyvm` and `deterministic-benchmark`
              run within token-scoped sandboxes that emit ledger entries and kernel logs.
            - **Audit-First Tooling** — shell modules (GPU access, module permissions,
              entropy auditing) share a common transcript pipeline that produces JSONL
              artifacts for downstream analysis.

            ## Phase 7 Commands

            The following administrative commands were introduced to finalize the
            Phase 7 release objectives:

            | Command | Purpose |
            | --- | --- |
            | `publish-shell-manual` | Regenerates this guide and records the publication in the snapshot ledger. |
            | `document-release-workflow` | Produces the release workflow reference for contributors. |
            | `document-module-tree` | Automates module tree documentation via the Roken Assembly manifest bridge. |
            | `kernel-ready-flag` | Marks the kernel as ready for the next evolution stage after documentation completes. |

            ## Deterministic Practices

            - Prefer scripted shell invocations via `cli/command_shell.py --script` so
              every task execution is replayable.
            - Use `deterministic-benchmark run --json` to capture reproducible
              validation telemetry before publishing artifacts.
            - Archive generated documentation under version control to maintain the
              signed audit trail across releases.
            """
        ).strip() + "\n"

        _write_text(target, body)
        try:
            event = self.snapshot_ledger.record_event(
                {
                    "kind": "documentation_published",
                    "artifact": "piyxu_deterministic_shell.md",
                    "timestamp": _isoformat_utc(datetime.now(timezone.utc)),
                }
            )
        except SnapshotLedgerError as exc:  # pragma: no cover - defensive guard
            raise DocumentationError(str(exc)) from exc
        return {"path": target, "event": event}

    def publish_release_workflow(self) -> Dict[str, object]:
        """Document the deterministic release workflow for contributors."""

        target = self.root / "docs" / "release_workflow.md"
        body = dedent(
            """
            # Deterministic Release Workflow

            This workflow describes how contributors produce a Codex-signed PIYXU OS2
            release while preserving the deterministic guarantees captured in the
            snapshot ledger.

            1. **Run validation suites** with `deterministic-benchmark run --json` and
               record the resulting ledger entry ID in the release checklist.
            2. **Publish documentation** by executing `publish-shell-manual` and
               `document-release-workflow` so the shell guide and workflow artifacts
               are refreshed from the repository root.
            3. **Export module inventory** through `document-module-tree --json` to
               capture the signed Roken Assembly manifest table for audit review.
            4. **Stage release metadata** by updating `README.md` with the phase
               summaries and confirming the roadmap anlatımı `docs/yol_hikayesi.md`
               içinde güncel tutulur.
            5. **Mark readiness** using `kernel-ready-flag --set` once the ledger
               reflects the publication events and deterministic benchmarks succeed.

            Each step emits an auditable record in `cli/data/snapshot_ledger.jsonl`
            so the release pipeline can verify the provenance of published binaries
            and documentation.
            """
        ).strip() + "\n"

        _write_text(target, body)
        try:
            event = self.snapshot_ledger.record_event(
                {
                    "kind": "release_workflow_documented",
                    "artifact": "release_workflow.md",
                    "timestamp": _isoformat_utc(datetime.now(timezone.utc)),
                }
            )
        except SnapshotLedgerError as exc:  # pragma: no cover - defensive guard
            raise DocumentationError(str(exc)) from exc
        return {"path": target, "event": event}

    def document_module_tree(self) -> Dict[str, object]:
        """Generate a markdown report of the registered shell modules."""

        modules_dir = self.root / "cli" / "modules"
        entries: List[Dict[str, object]] = []
        if modules_dir.exists():
            for path in sorted(modules_dir.glob("*.json")):
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                except json.JSONDecodeError as exc:
                    raise DocumentationError(
                        f"Invalid module manifest {path.name}: {exc}"
                    ) from exc
                entries.append(
                    {
                        "name": payload.get("name", path.stem),
                        "token_id": payload.get("token_id", ""),
                        "command_count": len(payload.get("commands", [])),
                        "path": path.relative_to(self.root).as_posix(),
                    }
                )

        lines = ["# Module Tree", "", "| Module | Token | Commands | Manifest |", "| --- | --- | ---:| --- |"]
        if entries:
            for entry in entries:
                lines.append(
                    "| {name} | {token_id} | {command_count} | `{path}` |".format(**entry)
                )
        else:
            lines.append("| _No manifests found_ | – | 0 | – |")

        body = "\n".join(lines) + "\n"
        target = self.root / "docs" / "module_tree.md"
        _write_text(target, body)
        try:
            event = self.snapshot_ledger.record_event(
                {
                    "kind": "module_tree_documented",
                    "artifact": "module_tree.md",
                    "module_count": len(entries),
                    "timestamp": _isoformat_utc(datetime.now(timezone.utc)),
                }
            )
        except SnapshotLedgerError as exc:  # pragma: no cover - defensive guard
            raise DocumentationError(str(exc)) from exc
        return {"path": target, "event": event, "modules": entries}

    def set_ready_flag(self, *, ready: bool) -> Dict[str, object]:
        """Persist the kernel readiness flag for the next evolution stage."""

        state_path = self.root / "cli" / "data" / "kernel_state.json"
        payload = {
            "ready_for_next_evolution": bool(ready),
            "updated_at": _isoformat_utc(datetime.now(timezone.utc)),
        }
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        try:
            event = self.snapshot_ledger.record_event(
                {
                    "kind": "kernel_ready_flag_updated",
                    "artifact": "kernel_state.json",
                    "ready": bool(ready),
                    "timestamp": payload["updated_at"],
                }
            )
        except SnapshotLedgerError as exc:  # pragma: no cover - defensive guard
            raise DocumentationError(str(exc)) from exc
        return {"path": state_path, "event": event, "ready": ready}


__all__ = ["DocumentationPublisher", "DocumentationError"]

