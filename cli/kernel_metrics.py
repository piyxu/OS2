"""Analytics helpers for extracting metrics from kernel logs."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Optional

SUCCESS_STATUSES = {"success", "succeeded", "completed", "ok", "done"}
FAILURE_STATUSES = {"failure", "failed", "error", "timeout", "crashed"}
SKIPPED_STATUSES = {"skipped", "ignored"}


@dataclass
class LabelSummary:
    total: int = 0
    success: int = 0
    failure: int = 0
    skipped: int = 0

    def register(self, status: str) -> None:
        normalized = status.lower()
        self.total += 1
        if normalized in SUCCESS_STATUSES:
            self.success += 1
        elif normalized in FAILURE_STATUSES:
            self.failure += 1
        elif normalized in SKIPPED_STATUSES:
            self.skipped += 1
        else:
            self.failure += 1

    @property
    def attempted(self) -> int:
        return self.success + self.failure

    def success_rate(self) -> float:
        attempted = self.attempted
        if attempted <= 0:
            return 0.0
        return self.success / attempted

    def to_dict(self) -> Dict[str, float | int]:
        return {
            "total": self.total,
            "success": self.success,
            "failure": self.failure,
            "skipped": self.skipped,
            "attempted": self.attempted,
            "success_rate": self.success_rate(),
        }


@dataclass
class TaskCompletionSummary:
    total: int
    attempted: int
    success: int
    failure: int
    skipped: int
    success_rate: float
    by_label: Dict[str, LabelSummary] = field(default_factory=dict)
    last_timestamp: Optional[int] = None
    last_sequence: Optional[int] = None

    def to_dict(self) -> Dict[str, object]:
        return {
            "total": self.total,
            "attempted": self.attempted,
            "success": self.success,
            "failure": self.failure,
            "skipped": self.skipped,
            "success_rate": self.success_rate,
            "by_label": {label: summary.to_dict() for label, summary in self.by_label.items()},
            "last_timestamp": self.last_timestamp,
            "last_sequence": self.last_sequence,
        }


class KernelTaskAnalyzer:
    """Parse kernel log events and measure task completion rates."""

    def __init__(self, path: Path) -> None:
        self._path = path

    @classmethod
    def for_workspace(cls, root: Path) -> "KernelTaskAnalyzer":
        return cls(root / "rust" / "os2-kernel" / "logs" / "kernel_events.jsonl")

    def _iter_events(self) -> Iterable[Dict[str, object]]:
        if not self._path.exists():
            return []
        try:
            lines = self._path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return []
        events = []
        for line in lines:
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(entry, dict):
                events.append(entry)
        return events

    def compute(self) -> TaskCompletionSummary:
        by_label: Dict[str, LabelSummary] = {}
        success = failure = skipped = 0
        last_timestamp: Optional[int] = None
        last_sequence: Optional[int] = None

        for event in self._iter_events():
            detail = event.get("detail")
            if not isinstance(detail, dict):
                continue
            status = detail.get("status")
            if not isinstance(status, str) or not status.strip():
                continue
            label = str(event.get("label") or event.get("kind") or "unknown")
            summary = by_label.setdefault(label, LabelSummary())
            summary.register(status)
            normalized = status.lower()
            if normalized in SUCCESS_STATUSES:
                success += 1
            elif normalized in FAILURE_STATUSES:
                failure += 1
            elif normalized in SKIPPED_STATUSES:
                skipped += 1
            else:
                failure += 1
            try:
                timestamp = int(event.get("timestamp"))
            except Exception:  # pragma: no cover - defensive
                timestamp = None
            try:
                sequence = int(event.get("sequence"))
            except Exception:  # pragma: no cover - defensive
                sequence = None
            if timestamp is not None:
                last_timestamp = timestamp if last_timestamp is None else max(last_timestamp, timestamp)
            if sequence is not None:
                last_sequence = sequence if last_sequence is None else max(last_sequence, sequence)

        total = success + failure + skipped
        attempted = success + failure
        success_rate = (success / attempted) if attempted > 0 else 0.0
        return TaskCompletionSummary(
            total=total,
            attempted=attempted,
            success=success,
            failure=failure,
            skipped=skipped,
            success_rate=success_rate,
            by_label=by_label,
            last_timestamp=last_timestamp,
            last_sequence=last_sequence,
        )


__all__ = [
    "KernelTaskAnalyzer",
    "LabelSummary",
    "TaskCompletionSummary",
]
