"""Ledger utilities for recording snapshot events."""

from __future__ import annotations

import hashlib
import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _isoformat_utc(dt: datetime) -> str:
    return dt.isoformat().replace("+00:00", "Z")


class SnapshotLedgerError(RuntimeError):
    """Raised when the snapshot ledger cannot append an event."""


class SnapshotLedger:
    """Append-only ledger used for snapshot-aware audit trails."""

    def __init__(self, path: Path, *, signing_key: str = "os2-snapshot-ledger") -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._signing_key = signing_key
        self._lock = threading.RLock()
        self._read_only = False

    def record_event(self, payload: Dict[str, object]) -> Dict[str, object]:
        """Append *payload* to the ledger and return the signed entry."""

        if self._read_only:
            raise SnapshotLedgerError("Snapshot ledger is read-only")
        event = dict(payload)
        event.setdefault("ts", _isoformat_utc(_now_utc()))
        serialized = json.dumps(event, sort_keys=True, ensure_ascii=False)
        event_id = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
        signature = hashlib.sha256((event_id + self._signing_key).encode("utf-8")).hexdigest()
        event["event_id"] = event_id
        event["signature"] = signature
        try:
            with self._lock:
                with self._path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(event, ensure_ascii=False) + "\n")
        except OSError as exc:  # pragma: no cover - defensive
            raise SnapshotLedgerError(str(exc)) from exc
        return event

    def set_read_only(self, value: bool) -> None:
        """Toggle the in-memory read-only guard for subsequent events."""

        with self._lock:
            self._read_only = bool(value)

    def is_read_only(self) -> bool:
        """Return ``True`` if the ledger rejects new events."""

        with self._lock:
            return self._read_only


__all__ = ["SnapshotLedger", "SnapshotLedgerError"]
