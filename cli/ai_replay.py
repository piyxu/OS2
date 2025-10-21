"""Replay and rollback utilities for deterministic AI executions."""

from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Iterable, List

from .model_registry import ModelRegistry
from .snapshot_ledger import SnapshotLedger

if TYPE_CHECKING:  # pragma: no cover
    from .ai_executor import AIExecutionResult
    from .llm_adapter import DeterministicLLMAdapter, InferenceResult


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _isoformat_utc(dt: datetime) -> str:
    return dt.isoformat().replace("+00:00", "Z")


class AIReplayError(RuntimeError):
    """Raised when replay or rollback data cannot be resolved."""


@dataclass(frozen=True)
class AIReplayRecord:
    """Represents a persisted AI execution available for replay."""

    event_id: str
    path: Path
    payload: Dict[str, object]

    @property
    def model(self) -> str:
        return str(self.payload.get("model", ""))

    @property
    def capability(self) -> str:
        return str(self.payload.get("capability", ""))

    @property
    def prompt(self) -> str:
        return str(self.payload.get("prompt", ""))

    @property
    def prompt_hash(self) -> str:
        return str(self.payload.get("prompt_hash", ""))

    @property
    def stored_at(self) -> str:
        return str(self.payload.get("stored_at", ""))


class AIReplayManager:
    """Persist model executions and offer deterministic replay hooks."""

    def __init__(
        self,
        root: Path,
        snapshot_ledger: SnapshotLedger,
        model_registry: ModelRegistry,
        llm_adapter: "DeterministicLLMAdapter",
    ) -> None:
        self._root = root
        self._snapshot_ledger = snapshot_ledger
        self._model_registry = model_registry
        self._llm_adapter = llm_adapter
        self._storage = self._root / "cli" / "ai_replay"
        self._storage.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    def _record_path(self, event_id: str) -> Path:
        return self._storage / f"{event_id}.json"

    def _write_payload(self, path: Path, payload: Dict[str, object]) -> None:
        serialised = json.dumps(payload, ensure_ascii=False, indent=2)
        path.write_text(serialised + "\n", encoding="utf-8")

    def _load_payload(self, path: Path) -> Dict[str, object]:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:  # pragma: no cover - defensive
            raise AIReplayError(f"Replay record missing at {path}") from exc
        except json.JSONDecodeError as exc:  # pragma: no cover - defensive
            raise AIReplayError(f"Replay record corrupt at {path}") from exc
        if not isinstance(raw, dict):
            raise AIReplayError(f"Replay record invalid format at {path}")
        return raw

    # ------------------------------------------------------------------
    def record_execution(self, result: "AIExecutionResult") -> AIReplayRecord:
        """Persist *result* for future deterministic replay."""

        payload: Dict[str, object] = {
            "event_id": result.registry_event_id,
            "model": result.record.name,
            "capability": result.record.capability,
            "prompt": result.prompt,
            "completion": result.completion,
            "prompt_hash": result.prompt_hash,
            "seed_hash": result.seed_hash,
            "deterministic_mode": result.deterministic_metadata,
            "verification": result.verification.to_metadata(),
            "runtime": result.runtime_plan.to_metadata(),
            "kernel_event": result.kernel_event,
            "gpu_event": result.gpu_event,
            "snapshot_event": result.snapshot_event,
            "context_switch": {
                "enter": result.context_switch_enter,
                "exit": result.context_switch_exit,
            },
            "hybrid_log": result.hybrid_log,
            "gpu_access": result.gpu_access,
            "feedback": result.feedback,
            "stored_at": _isoformat_utc(_now_utc()),
        }
        path = self._record_path(result.registry_event_id)
        with self._lock:
            self._write_payload(path, payload)
        return AIReplayRecord(event_id=result.registry_event_id, path=path, payload=payload)

    # ------------------------------------------------------------------
    def list_records(self) -> List[AIReplayRecord]:
        """Return all persisted replay records sorted by timestamp."""

        records: List[AIReplayRecord] = []
        with self._lock:
            for path in sorted(self._storage.glob("*.json")):
                payload = self._load_payload(path)
                event_id = str(payload.get("event_id", path.stem))
                records.append(AIReplayRecord(event_id=event_id, path=path, payload=payload))
        records.sort(key=lambda record: record.stored_at)
        return records

    def iter_records(self) -> Iterable[AIReplayRecord]:
        """Yield replay records without materialising the entire list."""

        with self._lock:
            for path in sorted(self._storage.glob("*.json")):
                payload = self._load_payload(path)
                event_id = str(payload.get("event_id", path.stem))
                yield AIReplayRecord(event_id=event_id, path=path, payload=payload)

    # ------------------------------------------------------------------
    def get_record(self, event_id: str) -> AIReplayRecord:
        path = self._record_path(event_id)
        payload = self._load_payload(path)
        return AIReplayRecord(event_id=event_id, path=path, payload=payload)

    # ------------------------------------------------------------------
    def replay(self, event_id: str) -> "InferenceResult":
        """Re-execute the deterministic inference for *event_id*."""

        record = self.get_record(event_id)
        model_name = record.model
        model = self._model_registry.get(model_name)
        if model is None:
            raise AIReplayError(f"Model '{model_name}' is not installed for replay")

        prompt = record.prompt
        snapshot_event = record.payload.get("snapshot_event", {})
        expected_digest = ""
        if isinstance(snapshot_event, dict):
            expected_digest = str(snapshot_event.get("response_digest", ""))

        seed_material = self._llm_adapter.seed_material(model, prompt)
        stored_seed_hash = str(record.payload.get("seed_hash", ""))
        computed_seed_hash = hashlib.sha256(seed_material.encode("utf-8")).hexdigest()
        if stored_seed_hash != computed_seed_hash:
            raise AIReplayError("Seed hash mismatch during replay")

        inference = self._llm_adapter.generate(model, prompt)
        if expected_digest and inference.digest != expected_digest:
            raise AIReplayError("Deterministic inference mismatch during replay")

        return inference

    # ------------------------------------------------------------------
    def rollback(self, event_id: str) -> Dict[str, object]:
        """Record a rollback event in the snapshot ledger for *event_id*."""

        record = self.get_record(event_id)
        snapshot_event = record.payload.get("snapshot_event")
        if not isinstance(snapshot_event, dict):
            raise AIReplayError("Snapshot metadata missing from replay record")

        payload = {
            "kind": "model_inference_rollback",
            "source_event_id": event_id,
            "model": record.model,
            "capability": record.capability,
            "prompt_hash": record.prompt_hash,
            "snapshot_event_id": snapshot_event.get("event_id"),
            "kernel_chain_hash": snapshot_event.get("kernel_chain_hash"),
            "hybrid_log": record.payload.get("hybrid_log"),
        }
        ledger_event = self._snapshot_ledger.record_event(payload)
        return ledger_event

    # ------------------------------------------------------------------
    def integrate_security_event(self, event: Dict[str, object]) -> Path:
        """Persist a security log entry alongside deterministic replay data."""

        payload = dict(event)
        payload.setdefault("event_type", "security")
        payload.setdefault("stored_at", _isoformat_utc(_now_utc()))
        event_id = str(
            payload.get("event_id")
            or hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
        )
        payload["event_id"] = event_id
        path = self._record_path(f"security-{event_id}")
        with self._lock:
            self._write_payload(path, payload)
        return path


__all__ = ["AIReplayManager", "AIReplayRecord", "AIReplayError"]

