"""Snapshot tagging utilities for Python VM sessions."""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence

from cli.snapshot_ledger import SnapshotLedger


@dataclass(frozen=True)
class SnapshotReservation:
    """Represents a reserved snapshot identifier for a Python VM session."""

    session_id: str
    snapshot_id: int


class PythonVMSnapshotRegistry:
    """Persist snapshot identifiers for Python VM sessions and emit ledger events."""

    def __init__(self, root: Path, ledger: SnapshotLedger) -> None:
        self._workspace = root.resolve()
        self._ledger = ledger
        self._lock = threading.RLock()
        self._root = self._workspace / "cli" / "python_vm" / "snapshots"
        self._root.mkdir(parents=True, exist_ok=True)
        self._sessions_dir = self._root / "sessions"
        self._sessions_dir.mkdir(parents=True, exist_ok=True)
        self._state_path = self._root / "state.json"
        self._state: Dict[str, object] = self._load_state()
        self._reservations: Dict[str, SnapshotReservation] = {}
        self._state_root = self._root / "state"
        self._state_root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    def _load_state(self) -> Dict[str, object]:
        try:
            payload = json.loads(self._state_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {"active_snapshot_id": 0, "next_snapshot_id": 1}
        except json.JSONDecodeError:
            return {"active_snapshot_id": 0, "next_snapshot_id": 1}
        active = int(payload.get("active_snapshot_id", 0))
        next_snapshot = int(payload.get("next_snapshot_id", max(active + 1, 1)))
        if next_snapshot <= 0:
            next_snapshot = 1
        return {"active_snapshot_id": active, "next_snapshot_id": next_snapshot}

    def _write_state(self) -> None:
        state_payload = {
            "active_snapshot_id": int(self._state.get("active_snapshot_id", 0)),
            "next_snapshot_id": int(self._state.get("next_snapshot_id", 1)),
        }
        with self._state_path.open("w", encoding="utf-8") as handle:
            json.dump(state_payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")

    # ------------------------------------------------------------------
    def reserve(self, session_id: str) -> SnapshotReservation:
        """Reserve and return a snapshot identifier for *session_id*."""

        with self._lock:
            next_snapshot = int(self._state.get("next_snapshot_id", 1))
            if next_snapshot <= 0:
                next_snapshot = 1
            reservation = SnapshotReservation(session_id=session_id, snapshot_id=next_snapshot)
            self._state["active_snapshot_id"] = reservation.snapshot_id
            self._state["next_snapshot_id"] = reservation.snapshot_id + 1
            self._write_state()
            self._reservations[session_id] = reservation
            return reservation

    # ------------------------------------------------------------------
    def record_tag(
        self,
        *,
        session_id: str,
        sandbox_id: str,
        mode: str,
        script: str,
        command_alias: Optional[str],
        script_args: Sequence[str],
        ledger_event_ids: Mapping[str, Optional[str]],
        resume_from: Optional[int] = None,
    ) -> Dict[str, object]:
        """Record a snapshot-tag event and persist session metadata."""

        with self._lock:
            reservation = self._reservations.get(session_id)
            if reservation is None:
                reservation = self.reserve(session_id)

            payload = {
                "kind": "python_vm_snapshot_tagged",
                "session_id": session_id,
                "sandbox_id": sandbox_id,
                "snapshot_id": reservation.snapshot_id,
                "mode": mode,
                "script": script,
                "command_alias": command_alias,
                "args": list(script_args),
                "resume_from_snapshot": resume_from,
                "ledger_event_ids": {k: v for k, v in ledger_event_ids.items() if v},
            }
            event = self._ledger.record_event(payload)

            session_payload = {
                "session_id": session_id,
                "sandbox_id": sandbox_id,
                "snapshot_id": reservation.snapshot_id,
                "event_id": event.get("event_id"),
                "ledger_event_ids": {k: v for k, v in ledger_event_ids.items() if v},
                "command_alias": command_alias,
                "mode": mode,
                "script": script,
                "args": list(script_args),
                "resume_from_snapshot": resume_from,
            }
            session_path = self._sessions_dir / f"{session_id}.json"
            with session_path.open("w", encoding="utf-8") as handle:
                json.dump(session_payload, handle, ensure_ascii=False, indent=2)
                handle.write("\n")

            self._reservations.pop(session_id, None)
            return event

    # ------------------------------------------------------------------
    def active_snapshot_id(self) -> int:
        """Return the most recently assigned snapshot identifier."""

        with self._lock:
            return int(self._state.get("active_snapshot_id", 0))

    # ------------------------------------------------------------------
    def cancel(self, session_id: str) -> None:
        """Release any reservation associated with *session_id*."""

        with self._lock:
            self._reservations.pop(session_id, None)

    # ------------------------------------------------------------------
    def state_path(self, snapshot_id: int) -> Path:
        """Return the state file path for *snapshot_id* without creating it."""

        sanitized = max(int(snapshot_id), 0)
        return self._state_root / f"{sanitized:08d}.json"

    def write_state(self, snapshot_id: int, state: Mapping[str, object]) -> Path:
        """Persist *state* for *snapshot_id* and return the path."""

        path = self.state_path(snapshot_id)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(state, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        return path

    def load_state(self, snapshot_id: int) -> Optional[Dict[str, object]]:
        """Return the stored state for *snapshot_id* if available."""

        path = self.state_path(snapshot_id)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
        if isinstance(payload, dict):
            return payload
        return None


__all__ = [
    "PythonVMSnapshotRegistry",
    "SnapshotReservation",
]
