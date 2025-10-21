"""Simple on-disk registry for OS2 CLI model management commands."""

from __future__ import annotations

import json
import hashlib
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, MutableSet, Optional


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _isoformat_utc(dt: datetime) -> str:
    return dt.isoformat().replace("+00:00", "Z")


class ModelRegistryError(Exception):
    """Raised when the registry encounters an invalid operation."""


@dataclass
class ModelRecord:
    """Represents a single registered model."""

    name: str
    source: str
    provider: str
    manifest: Optional[str]
    installed_at: str
    metadata: Dict[str, Any]
    capability: str = field(default="")

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ModelRecord":
        return cls(
            name=payload["name"],
            source=payload.get("source", ""),
            provider=payload.get("provider", ""),
            manifest=payload.get("manifest"),
            installed_at=payload.get("installed_at", _isoformat_utc(_now_utc())),
            metadata=dict(payload.get("metadata", {})),
            capability=payload.get("capability", ""),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "source": self.source,
            "provider": self.provider,
            "manifest": self.manifest,
            "installed_at": self.installed_at,
            "metadata": dict(self.metadata),
            "capability": self.capability,
        }


class ModelRegistry:
    """Maintains a local registry of models for CLI commands."""

    def __init__(self, storage_path: Path, *, signing_key: str = "os2-ledger-signing-key") -> None:
        self._storage_path = storage_path
        self._storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._models: Dict[str, ModelRecord] = {}
        self._ledger_path = self._storage_path.with_name("inference_ledger.jsonl")
        self._signing_key = signing_key
        self._capabilities: Dict[str, str] = {}
        self._load()

    # -------------------- persistence helpers --------------------
    def _load(self) -> None:
        if not self._storage_path.exists():
            return
        try:
            raw = json.loads(self._storage_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ModelRegistryError(f"Failed to read registry: {exc}") from exc
        if isinstance(raw, dict):
            records = raw.values()
        elif isinstance(raw, list):
            records = raw
        else:
            raise ModelRegistryError("Registry file must contain an object or array")
        for entry in records:
            record = ModelRecord.from_dict(entry)
            if not record.capability:
                record.capability = self._default_capability(record.name)
            self._models[record.name] = record
            self._capabilities[record.name] = record.capability

    def _persist(self) -> None:
        snapshot = [record.to_dict() for record in self._sorted_records()]
        temp_path = self._storage_path.with_suffix(".tmp")
        temp_path.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        temp_path.replace(self._storage_path)

    def _sorted_records(self) -> Iterable[ModelRecord]:
        return sorted(self._models.values(), key=lambda record: record.name.lower())

    def _default_capability(self, name: str) -> str:
        normalized = name.replace(" ", "-").lower()
        return f"cap.model.{normalized}"

    # -------------------- public API ------------------------------
    def list(self) -> List[ModelRecord]:
        with self._lock:
            return list(self._sorted_records())

    def get(self, name: str) -> Optional[ModelRecord]:
        with self._lock:
            return self._models.get(name)

    def install(
        self,
        name: str,
        *,
        source: str,
        provider: str,
        manifest: Optional[str],
        metadata: Optional[Dict[str, Any]] = None,
        overwrite: bool = False,
        capability: Optional[str] = None,
    ) -> ModelRecord:
        with self._lock:
            if not overwrite and name in self._models:
                raise ModelRegistryError(f"Model '{name}' is already installed")
            cap_name = capability or self._default_capability(name)
            record = ModelRecord(
                name=name,
                source=source,
                provider=provider,
                manifest=manifest,
                installed_at=_isoformat_utc(_now_utc()),
                metadata=dict(metadata or {}),
                capability=cap_name,
            )
            self._models[name] = record
            self._capabilities[name] = cap_name
            self._persist()
            return record

    def remove(self, name: str) -> ModelRecord:
        with self._lock:
            if name not in self._models:
                raise ModelRegistryError(f"Model '{name}' is not installed")
            record = self._models.pop(name)
            self._capabilities.pop(name, None)
            self._persist()
            return record

    def update_metadata(self, name: str, updates: Dict[str, Any], *, replace: bool = False) -> ModelRecord:
        """Update metadata for an installed model and persist the change."""

        with self._lock:
            if name not in self._models:
                raise ModelRegistryError(f"Model '{name}' is not installed")
            record = self._models[name]
            if replace:
                record.metadata = dict(updates)
            else:
                record.metadata.update(updates)
            self._persist()
            return record

    def capability_for(self, name: str) -> str:
        with self._lock:
            if name not in self._models:
                raise ModelRegistryError(f"Model '{name}' is not installed")
            return self._capabilities.get(name, self._default_capability(name))

    def sync_capabilities(self, target: MutableSet[str]) -> List[str]:
        with self._lock:
            added: List[str] = []
            for capability in self._capabilities.values():
                if capability not in target:
                    target.add(capability)
                    added.append(capability)
            return added

    def record_inference(
        self,
        name: str,
        prompt: str,
        response: str,
        *,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        capability = self.capability_for(name)
        base_event = {
            "model": name,
            "capability": capability,
            "prompt": prompt,
            "response": response,
            "metadata": dict(metadata or {}),
            "ts": _isoformat_utc(_now_utc()),
        }
        payload = json.dumps(base_event, sort_keys=True, ensure_ascii=False)
        event_id = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        signature = hashlib.sha256((event_id + self._signing_key).encode("utf-8")).hexdigest()
        event = {**base_event, "event_id": event_id, "signature": signature}
        with self._lock:
            self._ledger_path.parent.mkdir(parents=True, exist_ok=True)
            with self._ledger_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, ensure_ascii=False) + "\n")
        return event_id


__all__ = ["ModelRegistry", "ModelRegistryError", "ModelRecord"]
