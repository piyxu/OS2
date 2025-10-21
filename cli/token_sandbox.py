"""Token sandbox manager for Python VM sessions."""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence

from cli.snapshot_ledger import SnapshotLedger


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _isoformat_utc(dt: datetime) -> str:
    return dt.isoformat().replace("+00:00", "Z")


class TokenSandboxError(RuntimeError):
    """Raised when the sandbox manager encounters an error."""


class TokenSandboxBudgetExceeded(TokenSandboxError):
    """Raised when the sandbox token budget would be exceeded."""


@dataclass
class TokenSandbox:
    """Represents a single interpreter sandbox and its metadata."""

    manager: "TokenSandboxManager"
    sandbox_id: str
    path: Path
    metadata: Dict[str, object]

    creation_event: Optional[Dict[str, object]] = None
    release_event: Optional[Dict[str, object]] = None

    def __post_init__(self) -> None:
        self.path.mkdir(parents=True, exist_ok=True)
        self._metadata_path = self.path / "metadata.json"
        self._write_metadata()

    # ------------------------------------------------------------------
    def _write_metadata(self) -> None:
        payload = json.dumps(self.metadata, ensure_ascii=False, indent=2)
        self._metadata_path.write_text(payload + "\n", encoding="utf-8")

    def _update_metadata(self, updates: Dict[str, object]) -> None:
        with self.manager._lock:
            self.metadata.update(updates)
            self._write_metadata()

    # ------------------------------------------------------------------
    @property
    def token_budget(self) -> int:
        return int(self.metadata.get("token_budget", 0))

    @property
    def tokens_consumed(self) -> int:
        return int(self.metadata.get("tokens_consumed", 0))

    # ------------------------------------------------------------------
    def reserve(self, tokens: int) -> int:
        """Consume *tokens* from the sandbox budget."""

        if tokens <= 0:
            return self.tokens_consumed
        with self.manager._lock:
            budget = int(self.metadata.get("token_budget", 0))
            consumed = int(self.metadata.get("tokens_consumed", 0))
            new_total = consumed + int(tokens)
            if new_total > budget:
                raise TokenSandboxBudgetExceeded(
                    f"Token budget exceeded: {new_total} > {budget}"
                )
            reservations = list(self.metadata.get("reservations", []))
            reservations.append({"tokens": int(tokens), "ts": _isoformat_utc(_now_utc())})
            self.metadata["tokens_consumed"] = new_total
            self.metadata["reservations"] = reservations
            self._write_metadata()
            return new_total

    def finalize(self, status: str) -> Dict[str, object]:
        """Mark the sandbox as released and emit a ledger event."""

        with self.manager._lock:
            self.metadata["status"] = status
            self.metadata["released_at"] = _isoformat_utc(_now_utc())
            self._write_metadata()
        payload = {
            "kind": "python_vm_sandbox_released",
            "sandbox_id": self.sandbox_id,
            "session_id": self.metadata.get("session_id"),
            "status": status,
            "tokens_consumed": self.tokens_consumed,
        }
        snapshot_id = self.metadata.get("snapshot_id")
        if snapshot_id is not None:
            payload["snapshot_id"] = int(snapshot_id)
        event = self.manager.ledger.record_event(payload)
        self.release_event = event
        self._update_metadata({"release_event_id": event.get("event_id")})
        return event


class TokenSandboxManager:
    """Create and track token sandboxes for Python VM sessions."""

    def __init__(
        self,
        root: Path,
        ledger: SnapshotLedger,
        *,
        default_budget: int = 200,
    ) -> None:
        self._root = root
        self._root.mkdir(parents=True, exist_ok=True)
        self._ledger = ledger
        self._lock = threading.RLock()
        self._counter_path = self._root / "counter"
        self._counter = self._load_counter()
        self._default_budget = int(default_budget)

    # ------------------------------------------------------------------
    @property
    def ledger(self) -> SnapshotLedger:
        return self._ledger

    # ------------------------------------------------------------------
    def _load_counter(self) -> int:
        try:
            return int(self._counter_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return 0
        except ValueError:
            return 0

    def _next_id(self) -> str:
        with self._lock:
            self._counter += 1
            self._counter_path.write_text(str(self._counter), encoding="utf-8")
            return f"sandbox-{self._counter:05d}"

    # ------------------------------------------------------------------
    def create(
        self,
        session_id: str,
        *,
        token_budget: Optional[int] = None,
        capabilities: Optional[Sequence[str]] = None,
        snapshot_id: Optional[int] = None,
        resume_from_snapshot: Optional[int] = None,
        safe_mode: bool = False,
    ) -> TokenSandbox:
        budget = int(token_budget) if token_budget is not None else self._default_budget
        caps: Iterable[str] = sorted(set(capabilities or ()))
        sandbox_id = self._next_id()
        metadata: Dict[str, object] = {
            "sandbox_id": sandbox_id,
            "session_id": session_id,
            "created_at": _isoformat_utc(_now_utc()),
            "token_budget": budget,
            "tokens_consumed": 0,
            "capabilities": list(caps),
            "status": "active",
        }
        if snapshot_id is not None:
            metadata["snapshot_id"] = int(snapshot_id)
        if resume_from_snapshot is not None:
            metadata["resume_from_snapshot"] = int(resume_from_snapshot)
        if safe_mode:
            metadata["safe_mode"] = True
        sandbox = TokenSandbox(self, sandbox_id, self._root / sandbox_id, metadata)
        event_payload = {
            "kind": "python_vm_sandbox_created",
            "sandbox_id": sandbox_id,
            "session_id": session_id,
            "token_budget": budget,
            "capabilities": list(caps),
        }
        if snapshot_id is not None:
            event_payload["snapshot_id"] = int(snapshot_id)
        if resume_from_snapshot is not None:
            event_payload["resume_from_snapshot"] = int(resume_from_snapshot)
        if safe_mode:
            event_payload["safe_mode"] = True
        event = self._ledger.record_event(event_payload)
        sandbox.creation_event = event
        sandbox._update_metadata({"creation_event_id": event.get("event_id")})
        return sandbox


__all__ = [
    "TokenSandbox",
    "TokenSandboxError",
    "TokenSandboxBudgetExceeded",
    "TokenSandboxManager",
]
