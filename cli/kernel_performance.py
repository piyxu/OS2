"""Kernel performance monitoring utilities for energy, memory, and I/O metrics."""

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


def _isoformat(dt: datetime) -> str:
    return dt.isoformat().replace("+00:00", "Z")


class KernelPerformanceError(RuntimeError):
    """Raised when kernel performance metrics cannot be recorded or summarized."""


@dataclass
class PerformanceSample:
    sample_id: int
    recorded_at: str
    energy_joules: float
    memory_kb: int
    io_bytes: int
    component: str
    notes: Optional[str] = None
    ledger_event_id: Optional[str] = None
    ledger_signature: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "sample_id": self.sample_id,
            "recorded_at": self.recorded_at,
            "energy_joules": self.energy_joules,
            "memory_kb": self.memory_kb,
            "io_bytes": self.io_bytes,
            "component": self.component,
        }
        if self.notes:
            payload["notes"] = self.notes
        if self.ledger_event_id:
            payload["ledger_event_id"] = self.ledger_event_id
        if self.ledger_signature:
            payload["ledger_signature"] = self.ledger_signature
        return payload

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "PerformanceSample":
        sample_id = int(payload.get("sample_id", 0))
        recorded_at = str(payload.get("recorded_at", ""))
        energy_joules = float(payload.get("energy_joules", 0.0))
        memory_kb = int(payload.get("memory_kb", 0))
        io_bytes = int(payload.get("io_bytes", 0))
        component = str(payload.get("component", "kernel"))
        notes_value = payload.get("notes")
        notes = str(notes_value) if notes_value is not None else None
        ledger_event_id = payload.get("ledger_event_id")
        ledger_signature = payload.get("ledger_signature")
        return cls(
            sample_id=sample_id,
            recorded_at=recorded_at,
            energy_joules=energy_joules,
            memory_kb=memory_kb,
            io_bytes=io_bytes,
            component=component,
            notes=notes,
            ledger_event_id=str(ledger_event_id) if ledger_event_id else None,
            ledger_signature=str(ledger_signature) if ledger_signature else None,
        )


@dataclass
class _ComponentAggregate:
    samples: int = 0
    total_energy_joules: float = 0.0
    total_memory_kb: int = 0
    peak_memory_kb: int = 0
    total_io_bytes: int = 0
    peak_io_bytes: int = 0

    def register(self, sample: PerformanceSample) -> None:
        self.samples += 1
        self.total_energy_joules += sample.energy_joules
        self.total_memory_kb += sample.memory_kb
        self.total_io_bytes += sample.io_bytes
        if sample.memory_kb > self.peak_memory_kb:
            self.peak_memory_kb = sample.memory_kb
        if sample.io_bytes > self.peak_io_bytes:
            self.peak_io_bytes = sample.io_bytes

    def to_dict(self) -> Dict[str, object]:
        avg_energy = self.total_energy_joules / self.samples if self.samples else 0.0
        avg_memory = self.total_memory_kb / self.samples if self.samples else 0.0
        avg_io = self.total_io_bytes / self.samples if self.samples else 0.0
        energy_per_io = (
            self.total_energy_joules / self.total_io_bytes
            if self.total_io_bytes
            else 0.0
        )
        return {
            "samples": self.samples,
            "total_energy_joules": self.total_energy_joules,
            "total_memory_kb": self.total_memory_kb,
            "total_io_bytes": self.total_io_bytes,
            "average_energy_joules": avg_energy,
            "average_memory_kb": avg_memory,
            "average_io_bytes": avg_io,
            "peak_memory_kb": self.peak_memory_kb,
            "peak_io_bytes": self.peak_io_bytes,
            "energy_per_io_byte": energy_per_io,
        }


class KernelPerformanceMonitor:
    """Collects AI kernel performance samples and generates aggregate reports."""

    def __init__(self, root: Path, ledger: SnapshotLedger) -> None:
        self._root = root
        self._ledger = ledger
        self._path = root / "cli" / "data" / "kernel_performance.json"
        self._lock = threading.RLock()
        self._samples = self._load_or_initialize()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def list(self) -> List[Dict[str, object]]:
        with self._lock:
            return [sample.to_dict() for sample in self._samples]

    def record(
        self,
        energy_joules: float,
        memory_kb: int,
        io_bytes: int,
        *,
        component: str = "kernel",
        notes: str = "",
    ) -> Dict[str, object]:
        if energy_joules < 0:
            raise KernelPerformanceError("Energy must be non-negative")
        if memory_kb < 0:
            raise KernelPerformanceError("Memory must be non-negative")
        if io_bytes < 0:
            raise KernelPerformanceError("I/O bytes must be non-negative")
        component_text = component.strip()
        if not component_text:
            raise KernelPerformanceError("Component name cannot be empty")

        with self._lock:
            sample_id = self._next_id_locked()
            recorded_at = _isoformat(_now_utc())
            sample = PerformanceSample(
                sample_id=sample_id,
                recorded_at=recorded_at,
                energy_joules=float(energy_joules),
                memory_kb=int(memory_kb),
                io_bytes=int(io_bytes),
                component=component_text,
                notes=notes.strip() or None,
            )
            self._samples.append(sample)
            self._save_locked()

        ledger_event = self._ledger.record_event(
            {
                "kind": "kernel_performance_recorded",
                "sample_id": sample_id,
                "recorded_at": sample.recorded_at,
                "energy_joules": sample.energy_joules,
                "memory_kb": sample.memory_kb,
                "io_bytes": sample.io_bytes,
                "component": sample.component,
                "notes": sample.notes,
            }
        )

        with self._lock:
            sample.ledger_event_id = str(ledger_event.get("event_id"))
            sample.ledger_signature = str(ledger_event.get("signature"))
            self._samples[-1] = sample
            self._save_locked()

        return {"sample": sample.to_dict(), "ledger_event": ledger_event}

    def summary(self) -> Dict[str, object]:
        with self._lock:
            if not self._samples:
                return {
                    "samples": 0,
                    "total_energy_joules": 0.0,
                    "total_memory_kb": 0,
                    "total_io_bytes": 0,
                    "average_energy_joules": 0.0,
                    "average_memory_kb": 0.0,
                    "average_io_bytes": 0.0,
                    "peak_memory_kb": 0,
                    "peak_io_bytes": 0,
                    "energy_per_io_byte": 0.0,
                    "components": {},
                    "last_recorded_at": None,
                }

            total_energy = 0.0
            total_memory = 0
            total_io = 0
            peak_memory = 0
            peak_io = 0
            components: Dict[str, _ComponentAggregate] = {}
            last_recorded_at = ""

            for sample in self._samples:
                total_energy += sample.energy_joules
                total_memory += sample.memory_kb
                total_io += sample.io_bytes
                if sample.memory_kb > peak_memory:
                    peak_memory = sample.memory_kb
                if sample.io_bytes > peak_io:
                    peak_io = sample.io_bytes
                aggregate = components.setdefault(
                    sample.component, _ComponentAggregate()
                )
                aggregate.register(sample)
                if not last_recorded_at or sample.recorded_at > last_recorded_at:
                    last_recorded_at = sample.recorded_at

            sample_count = len(self._samples)
            average_energy = total_energy / sample_count
            average_memory = total_memory / sample_count
            average_io = total_io / sample_count
            energy_per_io = total_energy / total_io if total_io else 0.0

            return {
                "samples": sample_count,
                "total_energy_joules": total_energy,
                "total_memory_kb": total_memory,
                "total_io_bytes": total_io,
                "average_energy_joules": average_energy,
                "average_memory_kb": average_memory,
                "average_io_bytes": average_io,
                "peak_memory_kb": peak_memory,
                "peak_io_bytes": peak_io,
                "energy_per_io_byte": energy_per_io,
                "components": {
                    name: aggregate.to_dict() for name, aggregate in components.items()
                },
                "last_recorded_at": last_recorded_at,
            }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _load_or_initialize(self) -> List[PerformanceSample]:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if not self._path.exists():
            self._path.write_text("[]\n", encoding="utf-8")
            return []
        try:
            payload = json.loads(self._path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = []
        samples: List[PerformanceSample] = []
        if isinstance(payload, list):
            for item in payload:
                if isinstance(item, dict):
                    samples.append(PerformanceSample.from_dict(item))
        return samples

    def _save_locked(self) -> None:
        payload = [sample.to_dict() for sample in self._samples]
        self._path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    def _next_id_locked(self) -> int:
        if not self._samples:
            return 1
        return self._samples[-1].sample_id + 1


__all__ = [
    "KernelPerformanceMonitor",
    "KernelPerformanceError",
    "PerformanceSample",
]
