"""Kernel log writer utilities for recording Python VM activity."""

from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence


def _default_state() -> Dict[str, Any]:
    return {"timestamp": 0, "sequence": 0, "chain_hash": "0" * 64}


def _normalize_detail(detail: Mapping[str, Any]) -> Dict[str, Any]:
    """Return a JSON-serialisable copy of *detail*."""

    return json.loads(json.dumps(detail, ensure_ascii=False))


def _line_count(text: str) -> int:
    if not text:
        return 0
    count = text.count("\n")
    if not text.endswith("\n"):
        count += 1
    return count


@dataclass
class KernelLogWriter:
    """Append events compatible with the Rust kernel log format."""

    path: Path
    state_path: Path

    def __init__(self, path: Path, *, state_path: Optional[Path] = None) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if state_path is None:
            state_path = self.path.with_name(self.path.stem + "_state.json")
        self.state_path = state_path
        self._lock = threading.RLock()
        self._state = self._load_state()

    # ------------------------------------------------------------------
    @classmethod
    def for_workspace(cls, root: Path) -> "KernelLogWriter":
        log_path = root / "rust" / "os2-kernel" / "logs" / "kernel_events.jsonl"
        return cls(log_path)

    # ------------------------------------------------------------------
    def _load_state(self) -> Dict[str, Any]:
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return _default_state()
        except json.JSONDecodeError:
            return _default_state()

        state = _default_state()
        state.update(
            {
                "timestamp": int(payload.get("timestamp", 0)),
                "sequence": int(payload.get("sequence", 0)),
                "chain_hash": str(payload.get("chain_hash", "0" * 64)),
            }
        )
        return state

    def _write_state(self) -> None:
        self.state_path.write_text(
            json.dumps(self._state, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _token_id_from_session(session_id: str) -> int:
        digits = "".join(ch for ch in session_id if ch.isdigit())
        if not digits:
            return 0
        try:
            return int(digits)
        except ValueError:
            return 0

    @staticmethod
    def token_id_for_capability(capability: str) -> int:
        """Generate a deterministic token id for a capability string."""

        if not capability:
            return 0
        digits = "".join(ch for ch in capability if ch.isdigit())
        if digits:
            try:
                return int(digits[:18])
            except ValueError:
                pass
        digest = hashlib.sha256(capability.encode("utf-8")).hexdigest()
        return int(digest[:12], 16)

    # ------------------------------------------------------------------
    def record_event(
        self,
        *,
        kind: str,
        detail: Mapping[str, Any],
        label: Optional[str] = None,
        token_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Record an event compatible with ``KernelEvent::to_json``."""

        normalized_detail = _normalize_detail(detail)

        with self._lock:
            timestamp = self._state["timestamp"] + 1
            sequence = self._state["sequence"] + 1

            event: Dict[str, Any] = {
                "token_id": int(token_id or 0),
                "timestamp": timestamp,
                "sequence": sequence,
                "kind": kind,
                "detail": normalized_detail,
            }
            if label:
                event["label"] = label

            base_json = json.dumps(
                event, sort_keys=True, ensure_ascii=False, separators=(",", ":")
            )
            chain_hash = hashlib.sha256(
                (self._state["chain_hash"] + base_json).encode("utf-8")
            ).hexdigest()
            event["chain_hash"] = chain_hash

            self._state.update(
                {"timestamp": timestamp, "sequence": sequence, "chain_hash": chain_hash}
            )
            self._write_state()

            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, ensure_ascii=False) + "\n")

            return event

    def record_external_event(
        self,
        *,
        capability: str,
        source: str,
        detail: Mapping[str, Any],
        label: str = "external_event",
    ) -> Dict[str, Any]:
        """Record an external event attributed to a capability token."""

        payload = dict(detail)
        payload.setdefault("source", source)
        payload.setdefault("capability", capability)
        token_id = self.token_id_for_capability(capability)
        return self.record_event(
            kind="external",
            label=label,
            detail=payload,
            token_id=token_id,
        )

    # ------------------------------------------------------------------
    def record_context_switch(
        self,
        *,
        source: str,
        target: str,
        phase: str,
        detail: Mapping[str, Any],
        token_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Record a context-switch event between runtime components."""

        payload = dict(detail)
        payload.setdefault("source", source)
        payload.setdefault("target", target)
        payload.setdefault("phase", phase)
        return self.record_event(
            kind="context",
            label="context_switch",
            detail=payload,
            token_id=token_id,
        )

    # ------------------------------------------------------------------
    def record_python_stream(
        self,
        *,
        session_id: str,
        sandbox_id: str,
        stream: str,
        content: str,
        mode: str,
        command_alias: Optional[str],
        script: Optional[str],
        ledger_event_ids: Mapping[str, Optional[str]],
        token_budget: int,
        tokens_consumed: int,
        snapshot_id: int,
    ) -> Dict[str, Any]:
        detail = {
            "session_id": session_id,
            "sandbox_id": sandbox_id,
            "stream": stream,
            "content": content,
            "chars": len(content),
            "lines": _line_count(content),
            "is_empty": len(content) == 0,
            "mode": mode,
            "command_alias": command_alias,
            "script": script,
            "ledger_event_ids": {k: v for k, v in ledger_event_ids.items() if v},
            "token_budget": int(token_budget),
            "tokens_consumed": int(tokens_consumed),
            "snapshot_id": int(snapshot_id),
        }
        return self.record_event(
            kind="custom",
            label="python_vm_stream",
            detail=detail,
            token_id=self._token_id_from_session(session_id),
        )

    def record_python_session(
        self,
        *,
        session_id: str,
        sandbox_id: str,
        mode: str,
        command_alias: Optional[str],
        script: Optional[str],
        status: str,
        duration_ms: float,
        ledger_event_ids: Mapping[str, Optional[str]],
        stdout_event: Optional[Mapping[str, Any]],
        stderr_event: Optional[Mapping[str, Any]],
        token_budget: int,
        tokens_consumed: int,
        script_args: Sequence[str] = (),
        snapshot_id: int,
    ) -> Dict[str, Any]:
        detail = {
            "session_id": session_id,
            "sandbox_id": sandbox_id,
            "mode": mode,
            "command_alias": command_alias,
            "script": script,
            "args": list(script_args),
            "status": status,
            "duration_ms": round(duration_ms, 3),
            "ledger_event_ids": {k: v for k, v in ledger_event_ids.items() if v},
            "stdout_event_chain_hash": stdout_event.get("chain_hash") if stdout_event else None,
            "stderr_event_chain_hash": stderr_event.get("chain_hash") if stderr_event else None,
            "stdout_chars": stdout_event.get("detail", {}).get("chars") if stdout_event else 0,
            "stderr_chars": stderr_event.get("detail", {}).get("chars") if stderr_event else 0,
            "token_budget": int(token_budget),
            "tokens_consumed": int(tokens_consumed),
            "snapshot_id": int(snapshot_id),
        }

        return self.record_event(
            kind="custom",
            label="python_vm_session",
            detail=detail,
            token_id=self._token_id_from_session(session_id),
        )

    def record_python_async_task(
        self,
        *,
        session_id: str,
        sandbox_id: str,
        queue_id: str,
        task_id: str,
        name: str,
        status: str,
        result_repr: Optional[str],
        error: Optional[str],
        ledger_event_ids: Mapping[str, Optional[str]],
        snapshot_id: int,
    ) -> Dict[str, Any]:
        detail = {
            "session_id": session_id,
            "sandbox_id": sandbox_id,
            "queue_id": queue_id,
            "task_id": task_id,
            "name": name,
            "status": status,
            "result": result_repr,
            "error": error,
            "ledger_event_ids": {k: v for k, v in ledger_event_ids.items() if v},
            "snapshot_id": int(snapshot_id),
        }
        return self.record_event(
            kind="custom",
            label="python_vm_async_task",
            detail=detail,
            token_id=self._token_id_from_session(session_id),
        )


__all__ = ["KernelLogWriter"]

