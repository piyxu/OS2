"""Manage GPU access leases for deterministic AI executions."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

from .snapshot_ledger import SnapshotLedger


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _isoformat_utc(dt: datetime) -> str:
    return dt.isoformat().replace("+00:00", "Z")


class GPUAccessError(RuntimeError):
    """Raised when GPU access cannot be acquired or released."""


@dataclass
class GPUAccessLease:
    """Represents an acquired GPU lease."""

    lease_id: str
    capability: str
    backend: str
    device: str
    granted_ts: float
    granted_event: Dict[str, object]


class GPUAccessManager:
    """Coordinate GPU access leasing via the snapshot ledger."""

    def __init__(self, root: Path, snapshot_ledger: SnapshotLedger) -> None:
        self._root = root
        self._ledger = snapshot_ledger
        self._lock = threading.RLock()
        self._leases: Dict[str, GPUAccessLease] = {}
        self._log_path = self._root / "cli" / "data" / "gpu_access.jsonl"
        self._log_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    def _write_log_entry(self, entry: Dict[str, object]) -> None:
        serialized = json.dumps(entry, ensure_ascii=False)
        with self._log_path.open("a", encoding="utf-8") as handle:
            handle.write(serialized + "\n")

    def _lease_id(self, capability: str, backend: str, device: str, ts: float) -> str:
        payload = f"{capability}::{backend}::{device}::{ts:.6f}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------
    def acquire(
        self,
        *,
        capability: str,
        backend: str,
        device: str,
        model: Optional[str] = None,
    ) -> GPUAccessLease:
        """Acquire a lease for *device* and return the lease descriptor."""

        if backend not in {"cuda", "rocm", "mps"}:
            raise GPUAccessError(f"Backend '{backend}' is not GPU-enabled")

        now_ts = time.time()
        lease_id = self._lease_id(capability, backend, device, now_ts)
        granted_at = _isoformat_utc(_now_utc())

        payload = {
            "kind": "gpu_access_granted",
            "lease_id": lease_id,
            "capability": capability,
            "backend": backend,
            "device": device,
            "model": model,
            "granted_at": granted_at,
        }
        ledger_event = self._ledger.record_event(payload)

        log_entry = dict(ledger_event)
        log_entry.update({"event_type": "grant", "timestamp": granted_at})
        self._write_log_entry(log_entry)

        lease = GPUAccessLease(
            lease_id=lease_id,
            capability=capability,
            backend=backend,
            device=device,
            granted_ts=now_ts,
            granted_event=ledger_event,
        )
        with self._lock:
            self._leases[lease_id] = lease
        return lease

    # ------------------------------------------------------------------
    def release(
        self,
        lease_id: str,
        *,
        status: str = "completed",
        tokens: int = 0,
        detail: Optional[Dict[str, object]] = None,
    ) -> Dict[str, object]:
        """Release a previously acquired lease and record the event."""

        with self._lock:
            lease = self._leases.pop(lease_id, None)
        if lease is None:
            raise GPUAccessError(f"Unknown GPU lease '{lease_id}'")

        duration_ms = int((time.time() - lease.granted_ts) * 1000)
        payload = {
            "kind": "gpu_access_released",
            "lease_id": lease.lease_id,
            "capability": lease.capability,
            "backend": lease.backend,
            "device": lease.device,
            "duration_ms": duration_ms,
            "status": status,
            "tokens": int(tokens),
        }
        if detail:
            payload["detail"] = dict(detail)

        ledger_event = self._ledger.record_event(payload)
        log_entry = dict(ledger_event)
        log_entry.update(
            {
                "event_type": "release",
                "timestamp": ledger_event.get("ts", _isoformat_utc(_now_utc())),
                "duration_ms": duration_ms,
                "status": status,
                "tokens": int(tokens),
            }
        )
        self._write_log_entry(log_entry)
        return ledger_event

    # ------------------------------------------------------------------
    def list_leases(self) -> Dict[str, Dict[str, object]]:
        """Return a snapshot of active GPU leases."""

        with self._lock:
            return {
                lease_id: {
                    "capability": lease.capability,
                    "backend": lease.backend,
                    "device": lease.device,
                    "granted_ts": lease.granted_ts,
                    "granted_event": lease.granted_event,
                }
                for lease_id, lease in self._leases.items()
            }


__all__ = ["GPUAccessManager", "GPUAccessLease", "GPUAccessError"]

