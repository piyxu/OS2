"""Detect and prune unnecessary command modules to reduce entropy."""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from .snapshot_ledger import SnapshotLedger


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _isoformat_utc(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


class ModuleCleanerError(RuntimeError):
    """Raised when the module entropy workflow encounters an error."""


@dataclass
class ModuleInspection:
    """Represents the inspection result for a module definition file."""

    name: str
    path: str
    action: str
    reason: Optional[str]
    command_count: int

    def to_dict(self) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "name": self.name,
            "path": self.path,
            "action": self.action,
            "command_count": self.command_count,
        }
        if self.reason:
            payload["reason"] = self.reason
        return payload


class ModuleCleaner:
    """Detect and remove unnecessary command modules to reduce entropy."""

    def __init__(self, root: Path, ledger: SnapshotLedger) -> None:
        self._root = root.resolve()
        self._ledger = ledger
        self._modules_dir = self._root / "cli" / "modules"
        self._report_path = self._root / "cli" / "data" / "module_entropy_report.json"
        self._report_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def analyze(self) -> Dict[str, object]:
        """Inspect modules and return the current entropy report."""

        with self._lock:
            summary = self._evaluate_locked()
            self._save_report_locked(summary)
            return summary

    def prune(self) -> Dict[str, object]:
        """Remove unnecessary modules and record the ledger event."""

        with self._lock:
            summary = self._evaluate_locked()
            removable = [entry for entry in summary["modules"] if entry["action"] == "remove"]
            removed: List[Dict[str, object]] = []
            for entry in removable:
                module_path = self._root / entry["path"]
                if not module_path.exists():
                    continue
                try:
                    module_path.unlink()
                except OSError as exc:  # pragma: no cover - defensive
                    raise ModuleCleanerError(str(exc)) from exc
                removed.append(entry)
            result = {
                "generated_at": summary["generated_at"],
                "modules": summary["modules"],
                "removed": removed,
                "kept": [entry for entry in summary["modules"] if entry["action"] == "keep"],
                "counts": {
                    "total": summary["counts"]["total"],
                    "removable": summary["counts"]["removable"],
                    "removed": len(removed),
                },
            }
            if removed:
                ledger_event = self._ledger.record_event(
                    {
                        "kind": "module_entropy_pruned",
                        "removed": [
                            {
                                "name": entry["name"],
                                "path": entry["path"],
                                "reason": entry.get("reason"),
                                "command_count": entry.get("command_count", 0),
                            }
                            for entry in removed
                        ],
                    }
                )
                result["ledger_event"] = ledger_event
            self._save_report_locked(result)
            return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _evaluate_locked(self) -> Dict[str, object]:
        generated_at = _isoformat_utc(_now_utc())
        inspections: List[Dict[str, object]] = []
        if self._modules_dir.exists():
            for path in sorted(self._modules_dir.glob("*.json")):
                inspections.append(self._inspect_module(path).to_dict())
        removable = [entry for entry in inspections if entry["action"] == "remove"]
        kept = [entry for entry in inspections if entry["action"] == "keep"]
        summary = {
            "generated_at": generated_at,
            "modules": inspections,
            "removable": removable,
            "kept": kept,
            "counts": {
                "total": len(inspections),
                "removable": len(removable),
                "kept": len(kept),
            },
        }
        return summary

    def _inspect_module(self, path: Path) -> ModuleInspection:
        relative_path = path.relative_to(self._root)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return ModuleInspection(
                name=path.stem,
                path=str(relative_path),
                action="remove",
                reason="invalid-json",
                command_count=0,
            )
        if not isinstance(payload, dict):
            return ModuleInspection(
                name=path.stem,
                path=str(relative_path),
                action="remove",
                reason="invalid-structure",
                command_count=0,
            )
        name = str(payload.get("name") or path.stem)
        retain = payload.get("retain", True)
        commands = payload.get("commands", [])
        if retain is False:
            return ModuleInspection(
                name=name,
                path=str(relative_path),
                action="remove",
                reason="retain-flag-disabled",
                command_count=len(commands) if isinstance(commands, list) else 0,
            )
        if not isinstance(commands, list) or not commands:
            return ModuleInspection(
                name=name,
                path=str(relative_path),
                action="remove",
                reason="no-commands",
                command_count=0,
            )
        command_names: List[str] = []
        for command in commands:
            if not isinstance(command, dict):
                return ModuleInspection(
                    name=name,
                    path=str(relative_path),
                    action="remove",
                    reason="invalid-command-entry",
                    command_count=len(commands),
                )
            command_name = str(command.get("name", "")).strip()
            if not command_name:
                return ModuleInspection(
                    name=name,
                    path=str(relative_path),
                    action="remove",
                    reason="missing-command-name",
                    command_count=len(commands),
                )
            if command_name in command_names:
                return ModuleInspection(
                    name=name,
                    path=str(relative_path),
                    action="remove",
                    reason="duplicate-command-name",
                    command_count=len(commands),
                )
            command_names.append(command_name)
        return ModuleInspection(
            name=name,
            path=str(relative_path),
            action="keep",
            reason=None,
            command_count=len(command_names),
        )

    def _save_report_locked(self, report: Dict[str, object]) -> None:
        body = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
        self._report_path.write_text(body, encoding="utf-8")


__all__ = ["ModuleCleaner", "ModuleCleanerError", "ModuleInspection"]

