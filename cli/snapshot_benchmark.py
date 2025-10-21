"""Snapshot benchmark manager for periodic system evaluations."""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional

from .kernel_metrics import KernelTaskAnalyzer
from .snapshot_ledger import SnapshotLedger


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _isoformat(dt: datetime) -> str:
    return dt.isoformat().replace("+00:00", "Z")


def _parse_iso(text: str) -> datetime:
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    return datetime.fromisoformat(text)


EVALUATION_INTERVAL = timedelta(hours=24)


class SnapshotBenchmarkError(RuntimeError):
    """Raised when snapshot benchmark evaluation cannot proceed."""


@dataclass
class SnapshotBenchmarkRecord:
    benchmark_id: int
    started_at: str
    completed_at: str
    duration_ms: float
    task_summary: Dict[str, object]
    ledger_event_id: Optional[str] = None
    ledger_signature: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "benchmark_id": self.benchmark_id,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration_ms": self.duration_ms,
            "task_summary": self.task_summary,
        }
        if self.ledger_event_id:
            payload["ledger_event_id"] = self.ledger_event_id
        if self.ledger_signature:
            payload["ledger_signature"] = self.ledger_signature
        return payload

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "SnapshotBenchmarkRecord":
        benchmark_id = int(payload.get("benchmark_id", 0))
        started_at = str(payload.get("started_at", ""))
        completed_at = str(payload.get("completed_at", ""))
        duration_ms = float(payload.get("duration_ms", 0.0))
        summary = payload.get("task_summary", {})
        task_summary: Dict[str, object]
        if isinstance(summary, dict):
            task_summary = dict(summary)
        else:
            task_summary = {}
        record = cls(
            benchmark_id=benchmark_id,
            started_at=started_at,
            completed_at=completed_at,
            duration_ms=duration_ms,
            task_summary=task_summary,
            ledger_event_id=str(payload.get("ledger_event_id")) if payload.get("ledger_event_id") else None,
            ledger_signature=str(payload.get("ledger_signature")) if payload.get("ledger_signature") else None,
        )
        return record


class SnapshotBenchmarkManager:
    """Coordinates periodic benchmark evaluations against the snapshot ledger."""

    def __init__(
        self,
        root: Path,
        ledger: SnapshotLedger,
        analyzer_factory: Optional[Callable[[], KernelTaskAnalyzer]] = None,
    ) -> None:
        self._root = root
        self._ledger = ledger
        self._path = root / "cli" / "data" / "snapshot_benchmarks.json"
        self._lock = threading.RLock()
        self._records = self._load_or_initialize()
        self._analyzer_factory = analyzer_factory or (
            lambda: KernelTaskAnalyzer.for_workspace(self._root)
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def list(self) -> List[Dict[str, object]]:
        with self._lock:
            return [record.to_dict() for record in self._records]

    def status(self) -> Dict[str, object]:
        with self._lock:
            last = self._records[-1] if self._records else None
            next_allowed_at: Optional[str] = None
            seconds_until_next = 0.0
            if last:
                completed = _parse_iso(last.completed_at)
                next_time = completed + EVALUATION_INTERVAL
                next_allowed_at = _isoformat(next_time)
                remaining = (next_time - _now_utc()).total_seconds()
                if remaining > 0:
                    seconds_until_next = remaining
                else:
                    seconds_until_next = 0.0
            return {
                "last_benchmark": last.to_dict() if last else None,
                "next_allowed_at": next_allowed_at,
                "seconds_until_next": seconds_until_next,
            }

    def evaluate(self, *, force: bool = False) -> Dict[str, object]:
        start_wall = _now_utc()
        start_perf = time.perf_counter()
        with self._lock:
            self._ensure_interval_locked(start_wall, force=force)
            benchmark_id = self._next_id_locked()
        analyzer = self._analyzer_factory()
        summary = analyzer.compute().to_dict()
        duration_ms = (time.perf_counter() - start_perf) * 1000.0
        completed = _now_utc()
        record = SnapshotBenchmarkRecord(
            benchmark_id=benchmark_id,
            started_at=_isoformat(start_wall),
            completed_at=_isoformat(completed),
            duration_ms=duration_ms,
            task_summary=summary,
        )
        with self._lock:
            self._records.append(record)
            self._save_locked()
        ledger_event = self._ledger.record_event(
            {
                "kind": "snapshot_benchmark_completed",
                "benchmark_id": benchmark_id,
                "started_at": record.started_at,
                "completed_at": record.completed_at,
                "duration_ms": duration_ms,
                "task_summary": summary,
            }
        )
        record.ledger_event_id = ledger_event.get("event_id")  # type: ignore[assignment]
        record.ledger_signature = ledger_event.get("signature")  # type: ignore[assignment]
        with self._lock:
            self._records[-1] = record
            self._save_locked()
        return {"benchmark": record.to_dict(), "ledger_event": ledger_event}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _ensure_interval_locked(self, now: datetime, *, force: bool) -> None:
        if force:
            return
        if not self._records:
            return
        last = self._records[-1]
        completed = _parse_iso(last.completed_at)
        if now - completed < EVALUATION_INTERVAL:
            next_time = completed + EVALUATION_INTERVAL
            raise SnapshotBenchmarkError(
                "Snapshot benchmark already executed in the last 24 hours. "
                f"Next allowed at { _isoformat(next_time) }."
            )

    def _load_or_initialize(self) -> List[SnapshotBenchmarkRecord]:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if not self._path.exists():
            self._path.write_text("[]\n", encoding="utf-8")
            return []
        try:
            payload = json.loads(self._path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = []
        records: List[SnapshotBenchmarkRecord] = []
        if isinstance(payload, list):
            for item in payload:
                if isinstance(item, dict):
                    records.append(SnapshotBenchmarkRecord.from_dict(item))
        return records

    def _save_locked(self) -> None:
        payload = [record.to_dict() for record in self._records]
        self._path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    def _next_id_locked(self) -> int:
        if not self._records:
            return 1
        return max(record.benchmark_id for record in self._records) + 1


__all__ = [
    "SnapshotBenchmarkManager",
    "SnapshotBenchmarkError",
    "SnapshotBenchmarkRecord",
    "EVALUATION_INTERVAL",
]
