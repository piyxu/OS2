"""Manage the transition into the living deterministic system stage."""

from __future__ import annotations

import json
import math
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from .deterministic_recompile import DeterministicRecompileManager
from .kernel_performance import KernelPerformanceMonitor
from .module_cleaner import ModuleCleaner
from .self_feedback import SelfFeedbackAnalyzer
from .snapshot_benchmark import SnapshotBenchmarkManager
from .snapshot_ledger import SnapshotLedger, SnapshotLedgerError


_READY_STATES = {"ready", "stable", "clean"}
_HISTORY_LIMIT = 20


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _isoformat_utc(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def _deep_clone(payload: Dict[str, Any]) -> Dict[str, Any]:
    return json.loads(json.dumps(payload, ensure_ascii=False))


class LivingDeterministicSystemError(RuntimeError):
    """Raised when the living deterministic system manager cannot transition."""


class LivingDeterministicSystemManager:
    """Coordinate the kernel's transition into the living deterministic system stage."""

    def __init__(self, root: Path, ledger: SnapshotLedger) -> None:
        self._root = root.resolve()
        self._ledger = ledger
        self._path = self._root / "cli" / "data" / "living_system_state.json"
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._state = self._load_state()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def status(self, *, refresh: bool = False) -> Dict[str, Any]:
        with self._lock:
            state = _deep_clone(self._state)
        if not refresh:
            return state
        observation = self._collect_component_status()
        readiness = self._compute_readiness(observation)
        return {
            "state": state,
            "observation": {
                "components": observation,
                "readiness": readiness,
                "observed_at": _isoformat_utc(_now_utc()),
            },
        }

    def transition(
        self,
        operator: str,
        *,
        notes: str = "",
        force: bool = False,
    ) -> Dict[str, Any]:
        operator_text = operator.strip() or "unknown"
        notes_text = notes.strip() or None
        with self._lock:
            components = self._collect_component_status()
            readiness = self._compute_readiness(components)
            minimum_ready = readiness.get("minimum_ready", 0)
            ready_count = readiness.get("ready_components", 0)
            total = readiness.get("total_components", 0)
            if not force and (total == 0 or ready_count < minimum_ready):
                pending = readiness.get("pending_components", [])
                raise LivingDeterministicSystemError(
                    "Insufficient ready components for living transition: "
                    f"{ready_count}/{total} ready (minimum {minimum_ready}). "
                    f"Pending components: {', '.join(pending) if pending else 'none'}."
                )
            activated_at = _isoformat_utc(_now_utc())
            entry = {
                "stage": "living",
                "activated_at": activated_at,
                "operator": operator_text,
                "notes": notes_text,
                "components": components,
                "readiness": readiness,
            }
            try:
                ledger_event = self._ledger.record_event(
                    {
                        "kind": "living_system_transition",
                        "stage": "living",
                        "operator": operator_text,
                        "activated_at": activated_at,
                        "notes": notes_text,
                        "component_states": {
                            name: details.get("state", "unknown")
                            for name, details in components.items()
                        },
                        "ready_components": ready_count,
                        "total_components": total,
                        "minimum_ready": minimum_ready,
                        "pending_components": readiness.get("pending_components", []),
                    }
                )
            except SnapshotLedgerError as exc:  # pragma: no cover - defensive
                raise LivingDeterministicSystemError(str(exc)) from exc

            entry["ledger_event"] = ledger_event
            state = self._state
            state["current_stage"] = "living"
            state["activated_at"] = activated_at
            state["operator"] = operator_text
            state["notes"] = notes_text
            state["components"] = components
            state["readiness"] = readiness
            state["ledger_event"] = ledger_event
            history = state.setdefault("history", [])
            history.insert(0, entry)
            del history[_HISTORY_LIMIT:]
            self._save_locked()
            return _deep_clone(state)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _load_state(self) -> Dict[str, Any]:
        if not self._path.exists():
            default = {
                "current_stage": "dormant",
                "activated_at": None,
                "operator": None,
                "notes": None,
                "components": {},
                "readiness": {
                    "total_components": 0,
                    "ready_components": 0,
                    "ready_ratio": 0.0,
                    "minimum_ready": 0,
                    "pending_components": [],
                    "state": "unknown",
                },
                "ledger_event": None,
                "history": [],
            }
            self._path.write_text(json.dumps(default, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            return default
        try:
            payload = json.loads(self._path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = {}
        if not isinstance(payload, dict):
            payload = {}
        payload.setdefault("current_stage", "dormant")
        payload.setdefault("activated_at", None)
        payload.setdefault("operator", None)
        payload.setdefault("notes", None)
        payload.setdefault("components", {})
        payload.setdefault(
            "readiness",
            {
                "total_components": 0,
                "ready_components": 0,
                "ready_ratio": 0.0,
                "minimum_ready": 0,
                "pending_components": [],
                "state": "unknown",
            },
        )
        payload.setdefault("ledger_event", None)
        history = payload.get("history")
        if not isinstance(history, list):
            payload["history"] = []
        return payload

    def _save_locked(self) -> None:
        self._path.write_text(
            json.dumps(self._state, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    def _collect_component_status(self) -> Dict[str, Dict[str, Any]]:
        collectors: Iterable[Tuple[str, Any]] = (
            ("kernel_performance", self._collect_kernel_performance),
            ("snapshot_benchmarks", self._collect_snapshot_benchmarks),
            ("self_feedback", self._collect_self_feedback),
            ("module_entropy", self._collect_module_entropy),
            ("deterministic_recompile", self._collect_deterministic_recompile),
        )
        summary: Dict[str, Dict[str, Any]] = {}
        for name, collector in collectors:
            try:
                summary[name] = collector()
            except Exception as exc:  # pragma: no cover - defensive
                summary[name] = {"state": "error", "error": str(exc)}
        return summary

    def _collect_kernel_performance(self) -> Dict[str, Any]:
        monitor = KernelPerformanceMonitor(self._root, self._ledger)
        summary = monitor.summary()
        state = "ready" if summary.get("samples", 0) > 0 else "insufficient-data"
        return {"state": state, "summary": summary}

    def _collect_snapshot_benchmarks(self) -> Dict[str, Any]:
        manager = SnapshotBenchmarkManager(self._root, self._ledger)
        status = manager.status()
        state = "ready" if status.get("last_benchmark") else "awaiting-benchmark"
        return {
            "state": state,
            "status": status,
            "recent": manager.list()[-5:],
        }

    def _collect_self_feedback(self) -> Dict[str, Any]:
        analyzer = SelfFeedbackAnalyzer(self._root, self._ledger)
        summary = analyzer.summary()
        interactions = analyzer.recent_interactions(5)
        state = "ready" if summary.get("total_interactions", 0) > 0 else "awaiting-interactions"
        return {
            "state": state,
            "summary": summary,
            "recent": interactions,
        }

    def _collect_module_entropy(self) -> Dict[str, Any]:
        cleaner = ModuleCleaner(self._root, self._ledger)
        report = cleaner.analyze()
        counts = report.get("counts", {})
        removable = int(counts.get("removable", 0))
        state = "clean" if removable == 0 else "entropy-detected"
        return {
            "state": state,
            "report": report,
        }

    def _collect_deterministic_recompile(self) -> Dict[str, Any]:
        manager = DeterministicRecompileManager(self._root, self._ledger)
        pending = manager.pending()
        history = manager.history(limit=5)
        state = "stable" if not pending else "changes-pending"
        return {
            "state": state,
            "pending": pending,
            "recent_history": history,
        }

    def _compute_readiness(self, components: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        total = len(components)
        ready_components: List[str] = []
        pending_components: List[str] = []
        for name, details in components.items():
            state = str(details.get("state", "unknown"))
            if state in _READY_STATES:
                ready_components.append(name)
            else:
                pending_components.append(name)
        ready_count = len(ready_components)
        ready_ratio = (ready_count / total) if total else 0.0
        minimum_ready = 0
        if total:
            minimum_ready = min(total, max(3, math.ceil(total * 0.6)))
        if total == 0:
            overall_state = "unknown"
        elif ready_count >= total:
            overall_state = "ready"
        elif ready_count >= minimum_ready:
            overall_state = "stabilizing"
        elif ready_count > 0:
            overall_state = "incubating"
        else:
            overall_state = "dormant"
        return {
            "total_components": total,
            "ready_components": ready_count,
            "ready_ratio": ready_ratio,
            "minimum_ready": minimum_ready,
            "pending_components": pending_components,
            "state": overall_state,
            "ready_list": ready_components,
        }


__all__ = [
    "LivingDeterministicSystemError",
    "LivingDeterministicSystemManager",
]
