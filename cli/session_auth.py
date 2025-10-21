from __future__ import annotations

import json
import secrets
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

from cli.snapshot_ledger import SnapshotLedger, SnapshotLedgerError


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _isoformat(dt: datetime) -> str:
    return dt.isoformat().replace("+00:00", "Z")


class SessionAuthenticationError(RuntimeError):
    """Raised when the snapshot authentication workflow fails."""


class SnapshotAuthenticator:
    """Bind shell sessions to snapshot identities with ledger audit events."""

    def __init__(self, root: Path, ledger: SnapshotLedger) -> None:
        self._root = root.resolve()
        self._ledger = ledger
        self._lock = threading.RLock()
        self._path = self._root / "cli" / "data" / "session_auth.json"
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._state = self._load()

    def _load(self) -> Dict[str, object]:
        try:
            return json.loads(self._path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {"sessions": []}
        except json.JSONDecodeError:
            return {"sessions": []}

    def _write(self) -> None:
        with self._path.open("w", encoding="utf-8") as handle:
            json.dump(self._state, handle, ensure_ascii=False, indent=2)
            handle.write("\n")

    def authenticate(self, *, user: str, snapshot_id: int, reason: Optional[str] = None) -> Dict[str, object]:
        token = secrets.token_hex(16)
        payload = {
            "kind": "cli_snapshot_authenticated",
            "user": user,
            "snapshot_id": int(snapshot_id),
            "session_token": token,
            "reason": reason,
            "authenticated_at": _isoformat(_now_utc()),
        }

        try:
            ledger_event = self._ledger.record_event(payload)
        except SnapshotLedgerError as exc:
            raise SessionAuthenticationError(str(exc)) from exc

        record = {
            "user": user,
            "snapshot_id": int(snapshot_id),
            "session_token": token,
            "ledger_event_id": ledger_event.get("event_id"),
            "reason": reason,
            "authenticated_at": payload["authenticated_at"],
        }

        with self._lock:
            sessions = list(self._state.get("sessions", []))
            sessions.append(record)
            self._state["sessions"] = sessions
            self._write()

        return record


__all__ = ["SnapshotAuthenticator", "SessionAuthenticationError"]

