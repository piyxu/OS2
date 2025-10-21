"""Utilities for verifying deterministic Python VM session replay."""

from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Mapping, Optional

from .snapshot_ledger import SnapshotLedger


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _isoformat(dt: datetime) -> str:
    return dt.isoformat().replace("+00:00", "Z")


class PythonDeterminismError(RuntimeError):
    """Raised when deterministic replay validation cannot be completed."""


@dataclass
class VerificationReport:
    """Represents the replay status for a single Python VM session."""

    session_id: str
    snapshot_id: Optional[int]
    stdout_digest: str
    stderr_digest: str
    complete_event: Optional[str]
    stdout_chain_hash: Optional[str]
    stderr_chain_hash: Optional[str]
    errors: List[str]

    def to_dict(self) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "session_id": self.session_id,
            "snapshot_id": self.snapshot_id,
            "stdout_digest": self.stdout_digest,
            "stderr_digest": self.stderr_digest,
            "complete_event": self.complete_event,
        }
        if self.stdout_chain_hash:
            payload["stdout_chain_hash"] = self.stdout_chain_hash
        if self.stderr_chain_hash:
            payload["stderr_chain_hash"] = self.stderr_chain_hash
        if self.errors:
            payload["errors"] = list(self.errors)
        return payload


class PythonDeterminismVerifier:
    """Cross-check Python VM transcripts against ledger and kernel logs."""

    def __init__(self, root: Path, ledger: SnapshotLedger) -> None:
        self._root = root.resolve()
        self._ledger = ledger
        self._transcripts_root = (
            self._root / "rust" / "os2-kernel" / "logs" / "cli_sessions"
        )
        self._ledger_path = self._root / "cli" / "data" / "snapshot_ledger.jsonl"
        self._kernel_log_path = (
            self._root / "rust" / "os2-kernel" / "logs" / "kernel_events.jsonl"
        )
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    def verify(self, *, limit: Optional[int] = None) -> Dict[str, object]:
        """Validate deterministic replay for recorded Python VM sessions."""

        with self._lock:
            entries = self._load_transcripts(limit=limit)
            ledger_map = self._load_ledger()
            kernel_map = self._load_kernel_log()

            reports: List[VerificationReport] = []
            failures: List[Dict[str, object]] = []
            aggregate_stdout = hashlib.sha256()
            aggregate_stderr = hashlib.sha256()

            for payload in entries:
                report, ok = self._verify_entry(
                    payload, ledger_map=ledger_map, kernel_map=kernel_map
                )
                reports.append(report)
                aggregate_stdout.update(bytes.fromhex(report.stdout_digest))
                aggregate_stderr.update(bytes.fromhex(report.stderr_digest))
                if not ok:
                    failures.append(report.to_dict())

            summary = {
                "checked_sessions": len(reports),
                "verified_sessions": len(reports) - len(failures),
                "failed_sessions": len(failures),
                "aggregate_stdout_hash": aggregate_stdout.hexdigest()
                if reports
                else "0" * 64,
                "aggregate_stderr_hash": aggregate_stderr.hexdigest()
                if reports
                else "0" * 64,
                "reports": [report.to_dict() for report in reports],
                "failures": failures,
                "verified_at": _isoformat(_now_utc()),
            }

            ledger_event = self._ledger.record_event(
                {
                    "kind": "python_vm_replay_verified",
                    "checked_sessions": summary["checked_sessions"],
                    "verified_sessions": summary["verified_sessions"],
                    "failed_sessions": summary["failed_sessions"],
                    "aggregate_stdout_hash": summary["aggregate_stdout_hash"],
                    "aggregate_stderr_hash": summary["aggregate_stderr_hash"],
                }
            )
            summary["ledger_event"] = ledger_event
            return summary

    # ------------------------------------------------------------------
    def _load_transcripts(
        self, *, limit: Optional[int]
    ) -> List[Dict[str, object]]:
        if not self._transcripts_root.exists():
            return []

        transcripts: List[Dict[str, object]] = []
        files = sorted(self._transcripts_root.glob("session-*.jsonl"))
        for path in files:
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except OSError:  # pragma: no cover - defensive fallback
                continue
            for line in lines:
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if payload.get("type") == "python_vm":
                    transcripts.append(payload)
        if limit is not None and limit >= 0:
            return transcripts[-int(limit) :] if transcripts else []
        return transcripts

    def _load_ledger(self) -> Dict[str, Dict[str, object]]:
        if not self._ledger_path.exists():
            return {}
        data: Dict[str, Dict[str, object]] = {}
        for line in self._ledger_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            event_id = str(payload.get("event_id", ""))
            if event_id:
                data[event_id] = payload
        return data

    def _load_kernel_log(self) -> Dict[str, Dict[str, object]]:
        if not self._kernel_log_path.exists():
            return {}
        events: Dict[str, Dict[str, object]] = {}
        for line in self._kernel_log_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            chain_hash = str(payload.get("chain_hash", ""))
            if chain_hash:
                events[chain_hash] = payload
        return events

    def _verify_entry(
        self,
        payload: Mapping[str, object],
        *,
        ledger_map: Dict[str, Dict[str, object]],
        kernel_map: Dict[str, Dict[str, object]],
    ) -> tuple[VerificationReport, bool]:
        session_id = str(payload.get("session_id", ""))
        snapshot_id = payload.get("snapshot_id")
        try:
            snapshot = int(snapshot_id) if snapshot_id is not None else None
        except (TypeError, ValueError):
            snapshot = None

        stdout = str(payload.get("stdout", ""))
        stderr = str(payload.get("stderr", ""))
        stdout_digest = hashlib.sha256(stdout.encode("utf-8")).hexdigest()
        stderr_digest = hashlib.sha256(stderr.encode("utf-8")).hexdigest()

        events = payload.get("events")
        if isinstance(events, Mapping):
            complete_event_id = events.get("complete")
        else:
            complete_event_id = None
        complete_event = str(complete_event_id) if complete_event_id else None

        errors: List[str] = []
        ledger_entry: Optional[Dict[str, object]] = None
        if complete_event:
            ledger_entry = ledger_map.get(complete_event)
            if ledger_entry is None:
                errors.append("missing_complete_event")
            else:
                if int(ledger_entry.get("stdout_len", len(stdout))) != len(stdout):
                    errors.append("stdout_length_mismatch")
                if int(ledger_entry.get("stderr_len", len(stderr))) != len(stderr):
                    errors.append("stderr_length_mismatch")
        else:
            errors.append("missing_complete_event")

        kernel_refs = payload.get("kernel_log") if isinstance(payload, Mapping) else None
        stdout_chain_hash: Optional[str] = None
        stderr_chain_hash: Optional[str] = None

        if isinstance(kernel_refs, Mapping):
            stdout_ref = kernel_refs.get("stdout")
            if isinstance(stdout_ref, Mapping):
                stdout_chain_hash = str(stdout_ref.get("chain_hash")) or None
                if stdout_chain_hash:
                    event = kernel_map.get(stdout_chain_hash)
                    if not event:
                        errors.append("missing_kernel_stdout")
                    else:
                        content = str(event.get("detail", {}).get("content", ""))
                        if content != stdout:
                            errors.append("stdout_content_mismatch")
            stderr_ref = kernel_refs.get("stderr")
            if isinstance(stderr_ref, Mapping):
                stderr_chain_hash = str(stderr_ref.get("chain_hash")) or None
                if stderr_chain_hash:
                    event = kernel_map.get(stderr_chain_hash)
                    if not event:
                        errors.append("missing_kernel_stderr")
                    else:
                        content = str(event.get("detail", {}).get("content", ""))
                        if content != stderr:
                            errors.append("stderr_content_mismatch")

        report = VerificationReport(
            session_id=session_id,
            snapshot_id=snapshot,
            stdout_digest=stdout_digest,
            stderr_digest=stderr_digest,
            complete_event=complete_event,
            stdout_chain_hash=stdout_chain_hash,
            stderr_chain_hash=stderr_chain_hash,
            errors=errors,
        )
        return report, not errors


__all__ = [
    "PythonDeterminismError",
    "PythonDeterminismVerifier",
    "VerificationReport",
]
