"""Deterministic recompile approval workflow."""

from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, MutableMapping, Optional, Sequence

from .snapshot_ledger import SnapshotLedger, SnapshotLedgerError


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _isoformat_utc(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def _hash_file(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            sha.update(chunk)
    return sha.hexdigest()


@dataclass(frozen=True)
class RecompileRecord:
    """Represents the deterministic snapshot of a queued change."""

    submission_id: int
    change_id: str
    description: str
    paths: List[str]
    hashes: Dict[str, str]
    submitted_by: str
    queued_at: str
    tags: List[str]

    def to_dict(self) -> Dict[str, object]:
        return {
            "submission_id": self.submission_id,
            "change_id": self.change_id,
            "description": self.description,
            "paths": list(self.paths),
            "hashes": dict(self.hashes),
            "submitted_by": self.submitted_by,
            "queued_at": self.queued_at,
            "tags": list(self.tags),
        }


class DeterministicRecompileError(RuntimeError):
    """Raised when the deterministic recompile workflow fails."""


class DeterministicRecompileManager:
    """Approve code changes by hashing files and verifying them deterministically."""

    def __init__(
        self,
        root: Path,
        ledger: SnapshotLedger,
        *,
        max_history: int = 100,
    ) -> None:
        self._root = root.resolve()
        self._ledger = ledger
        self._path = self._root / "cli" / "data" / "deterministic_recompile.json"
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._max_history = max(1, max_history)
        self._lock = threading.RLock()
        self._state = self._load_state()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def queue(
        self,
        change_id: str,
        paths: Sequence[str],
        *,
        description: str = "",
        submitted_by: str = "unknown",
        tags: Optional[Sequence[str]] = None,
    ) -> Dict[str, object]:
        change_id = change_id.strip()
        if not change_id:
            raise DeterministicRecompileError("Change ID is required")
        normalized_paths = self._normalize_paths(paths)
        if not normalized_paths:
            raise DeterministicRecompileError("At least one --path must be provided")
        tag_list = sorted({tag.strip() for tag in tags or [] if tag and tag.strip()})
        with self._lock:
            if any(entry["change_id"] == change_id for entry in self._state["pending"]):
                raise DeterministicRecompileError(f"Change '{change_id}' is already pending")
            submission_id = int(self._state.get("next_submission_id", 1))
            queued_at = _isoformat_utc(_now_utc())
            hashes = self._hash_paths(normalized_paths)
            record = RecompileRecord(
                submission_id=submission_id,
                change_id=change_id,
                description=description.strip(),
                paths=normalized_paths,
                hashes=hashes,
                submitted_by=submitted_by,
                queued_at=queued_at,
                tags=tag_list,
            )
            event_payload = {
                "kind": "deterministic_recompile_submitted",
                "change_id": change_id,
                "submission_id": submission_id,
                "paths": normalized_paths,
                "hashes": hashes,
                "description": record.description,
                "submitted_by": submitted_by,
                "tags": tag_list,
            }
            try:
                ledger_event = self._ledger.record_event(event_payload)
            except SnapshotLedgerError as exc:  # pragma: no cover - defensive
                raise DeterministicRecompileError(str(exc)) from exc
            payload = record.to_dict()
            payload["ledger_event"] = ledger_event
            payload["last_outcome"] = None
            self._state["pending"].append(payload)
            self._state["next_submission_id"] = submission_id + 1
            self._save_locked()
            return {"queued_change": payload}

    def pending(self) -> List[Dict[str, object]]:
        with self._lock:
            return [self._clone(entry) for entry in self._state["pending"]]

    def history(self, limit: Optional[int] = None) -> List[Dict[str, object]]:
        with self._lock:
            records = [self._clone(entry) for entry in self._state["history"]]
            if limit is None or limit >= len(records):
                return records
            return records[: max(0, int(limit))]

    def approve(
        self,
        change_id: str,
        *,
        reviewer: str = "unknown",
    ) -> Dict[str, object]:
        change_id = change_id.strip()
        if not change_id:
            raise DeterministicRecompileError("Change ID is required")
        with self._lock:
            index = self._find_pending_index(change_id)
            if index is None:
                raise DeterministicRecompileError(
                    f"Change '{change_id}' is not pending deterministic approval"
                )
            entry = self._state["pending"][index]
            observed_hashes = self._hash_paths(entry["paths"], allow_missing=True)
            mismatches = self._detect_mismatches(entry["hashes"], observed_hashes)
            checked_at = _isoformat_utc(_now_utc())
            if mismatches:
                outcome = {
                    "status": "mismatch",
                    "details": mismatches,
                    "observed_hashes": observed_hashes,
                    "checked_at": checked_at,
                }
                try:
                    ledger_event = self._ledger.record_event(
                        {
                            "kind": "deterministic_recompile_mismatch",
                            "change_id": change_id,
                            "submission_id": entry["submission_id"],
                            "mismatches": mismatches,
                            "observed_hashes": observed_hashes,
                            "reviewer": reviewer,
                        }
                    )
                except SnapshotLedgerError as exc:  # pragma: no cover - defensive
                    raise DeterministicRecompileError(str(exc)) from exc
                outcome["ledger_event"] = ledger_event
                entry["last_outcome"] = outcome
                entry["last_checked_at"] = checked_at
                self._save_locked()
                raise DeterministicRecompileError(
                    "Deterministic recompile mismatch detected"
                )
            try:
                ledger_event = self._ledger.record_event(
                    {
                        "kind": "deterministic_recompile_approved",
                        "change_id": change_id,
                        "submission_id": entry["submission_id"],
                        "paths": entry["paths"],
                        "hashes": entry["hashes"],
                        "reviewer": reviewer,
                        "queued_at": entry["queued_at"],
                        "approved_at": checked_at,
                    }
                )
            except SnapshotLedgerError as exc:  # pragma: no cover - defensive
                raise DeterministicRecompileError(str(exc)) from exc
            approved_record = {
                "submission_id": entry["submission_id"],
                "change_id": change_id,
                "description": entry.get("description", ""),
                "paths": list(entry["paths"]),
                "hashes": dict(entry["hashes"]),
                "submitted_by": entry.get("submitted_by", "unknown"),
                "queued_at": entry.get("queued_at"),
                "approved_at": checked_at,
                "reviewer": reviewer,
                "tags": list(entry.get("tags", [])),
                "ledger_event": ledger_event,
            }
            del self._state["pending"][index]
            self._state["history"].insert(0, approved_record)
            if len(self._state["history"]) > self._max_history:
                del self._state["history"][self._max_history :]
            self._save_locked()
            return {"approved_change": approved_record}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _load_state(self) -> MutableMapping[str, object]:
        state: MutableMapping[str, object] = {
            "pending": [],
            "history": [],
            "next_submission_id": 1,
        }
        if not self._path.exists():
            return state
        try:
            payload = json.loads(self._path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return state
        if not isinstance(payload, dict):
            return state
        for key in ("pending", "history", "next_submission_id"):
            if key in payload:
                state[key] = payload[key]
        pending = state.get("pending", [])
        if isinstance(pending, list):
            normalized_pending = []
            for item in pending:
                if isinstance(item, dict):
                    normalized_pending.append(item)
            state["pending"] = normalized_pending
        else:
            state["pending"] = []
        history = state.get("history", [])
        if isinstance(history, list):
            normalized_history = []
            for item in history:
                if isinstance(item, dict):
                    normalized_history.append(item)
            state["history"] = normalized_history
        else:
            state["history"] = []
        try:
            state["next_submission_id"] = int(state.get("next_submission_id", 1))
        except (TypeError, ValueError):
            state["next_submission_id"] = 1
        return state

    def _save_locked(self) -> None:
        with self._path.open("w", encoding="utf-8") as handle:
            json.dump(self._state, handle, indent=2, ensure_ascii=False)

    def _normalize_paths(self, paths: Sequence[str]) -> List[str]:
        normalized: List[str] = []
        for raw in paths:
            if raw is None:
                continue
            candidate = str(raw).strip()
            if not candidate:
                continue
            resolved = (self._root / candidate).resolve()
            if not str(resolved).startswith(str(self._root)):
                raise DeterministicRecompileError(
                    f"Path '{candidate}' escapes the workspace sandbox"
                )
            if not resolved.exists() or not resolved.is_file():
                raise DeterministicRecompileError(
                    f"Path '{candidate}' does not reference a tracked file"
                )
            normalized.append(str(resolved.relative_to(self._root)))
        return sorted(dict.fromkeys(normalized))

    def _hash_paths(
        self, paths: Iterable[str], *, allow_missing: bool = False
    ) -> Dict[str, Optional[str]]:
        hashes: Dict[str, Optional[str]] = {}
        for rel_path in paths:
            resolved = (self._root / rel_path).resolve()
            if not str(resolved).startswith(str(self._root)):
                hashes[rel_path] = None
                continue
            if not resolved.exists() or not resolved.is_file():
                if allow_missing:
                    hashes[rel_path] = None
                    continue
                raise DeterministicRecompileError(
                    f"Path '{rel_path}' does not reference a tracked file"
                )
            hashes[rel_path] = _hash_file(resolved)
        return hashes

    def _detect_mismatches(
        self,
        expected: Dict[str, str],
        observed: Dict[str, Optional[str]],
    ) -> List[str]:
        mismatches: List[str] = []
        for path, expected_hash in expected.items():
            observed_hash = observed.get(path)
            if observed_hash is None:
                mismatches.append(f"missing:{path}")
                continue
            if observed_hash != expected_hash:
                mismatches.append(f"hash:{path}")
        return mismatches

    def _find_pending_index(self, change_id: str) -> Optional[int]:
        for index, entry in enumerate(self._state["pending"]):
            if entry.get("change_id") == change_id:
                return index
        return None

    def _clone(self, payload: Dict[str, object]) -> Dict[str, object]:
        cloned = dict(payload)
        for key in ("paths", "tags"):
            if key in cloned and isinstance(cloned[key], list):
                cloned[key] = list(cloned[key])
        if "hashes" in cloned and isinstance(cloned["hashes"], dict):
            cloned["hashes"] = dict(cloned["hashes"])
        return cloned


__all__ = [
    "DeterministicRecompileError",
    "DeterministicRecompileManager",
]
