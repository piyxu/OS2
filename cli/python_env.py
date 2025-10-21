"""Deterministic Python environment management for the PIYXU shell."""

from __future__ import annotations

import hashlib
import json
import platform
import re
import shutil
import threading
import time
import venv
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

from cli.snapshot_ledger import SnapshotLedger


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _isoformat_utc(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4096), b""):
            digest.update(chunk)
    return digest.hexdigest()


class PythonEnvironmentError(RuntimeError):
    """Raised when the environment manager encounters an error."""


class InvalidPythonEnvironmentName(PythonEnvironmentError):
    """Raised when an environment name fails validation."""


class PythonEnvironmentExistsError(PythonEnvironmentError):
    """Raised when attempting to recreate an existing environment."""


@dataclass
class PythonEnvironmentInfo:
    """Structured metadata describing a managed Python environment."""

    environment_id: str
    path: Path
    python_executable: Path
    python_version: str
    created_at: str
    status: str
    with_pip: bool
    ledger_event_ids: Dict[str, str] = field(default_factory=dict)
    metadata_path: Optional[Path] = None
    description: Optional[str] = None
    pyvenv_cfg_hash: Optional[str] = None
    duration_ms: Optional[float] = None

    def to_dict(self) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "environment_id": self.environment_id,
            "path": str(self.path),
            "python_executable": str(self.python_executable),
            "python_version": self.python_version,
            "created_at": self.created_at,
            "status": self.status,
            "with_pip": self.with_pip,
            "ledger_event_ids": dict(self.ledger_event_ids),
        }
        if self.metadata_path is not None:
            payload["metadata_path"] = str(self.metadata_path)
        if self.description is not None:
            payload["description"] = self.description
        if self.pyvenv_cfg_hash is not None:
            payload["pyvenv_cfg_hash"] = self.pyvenv_cfg_hash
        if self.duration_ms is not None:
            payload["duration_ms"] = round(float(self.duration_ms), 3)
        return payload


class PythonEnvironmentManager:
    """Create and track deterministic Python environments."""

    _NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")

    def __init__(self, workspace_root: Path, ledger: SnapshotLedger) -> None:
        self._workspace_root = workspace_root.resolve()
        self._environments_root = self._workspace_root / "cli" / "python_vm" / "environments"
        self._environments_root.mkdir(parents=True, exist_ok=True)
        self._ledger = ledger
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    def _sanitize_name(self, name: str) -> str:
        candidate = name.strip()
        if not candidate:
            raise InvalidPythonEnvironmentName("Environment name cannot be empty")
        if not self._NAME_PATTERN.match(candidate):
            raise InvalidPythonEnvironmentName(
                "Environment name must match [A-Za-z0-9][A-Za-z0-9_-]{0,63}"
            )
        return candidate

    def _relative(self, path: Path) -> Path:
        resolved = path.resolve()
        try:
            return resolved.relative_to(self._workspace_root)
        except ValueError:
            return resolved

    def _metadata_path(self, environment_id: str) -> Path:
        return self._environments_root / environment_id / "metadata.json"

    def _write_metadata(self, environment_id: str, payload: Dict[str, object]) -> Path:
        metadata_path = self._metadata_path(environment_id)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return metadata_path

    def _discover_python_executable(self, env_path: Path) -> Path:
        candidates = [
            env_path / "bin" / "python",
            env_path / "bin" / "python3",
            env_path / "Scripts" / "python.exe",
            env_path / "Scripts" / "python",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise PythonEnvironmentError(f"Unable to locate python executable in {env_path}")

    # ------------------------------------------------------------------
    def create(
        self,
        name: str,
        *,
        requested_by: str,
        description: Optional[str] = None,
        with_pip: bool = True,
    ) -> PythonEnvironmentInfo:
        environment_id = self._sanitize_name(name)
        env_path = self._environments_root / environment_id
        created_at = _isoformat_utc(_now_utc())
        start_time = time.time()

        with self._lock:
            if env_path.exists():
                failure_payload = {
                    "kind": "python_vm_env_creation_failed",
                    "environment_id": environment_id,
                    "requested_by": requested_by,
                    "status": "exists",
                }
                if description:
                    failure_payload["description"] = description
                self._ledger.record_event(failure_payload)
                raise PythonEnvironmentExistsError(f"Environment already exists: {environment_id}") from None

            start_payload = {
                "kind": "python_vm_env_creation_started",
                "environment_id": environment_id,
                "requested_by": requested_by,
                "with_pip": bool(with_pip),
            }
            if description:
                start_payload["description"] = description
            start_event = self._ledger.record_event(start_payload)

            metadata: Dict[str, object] = {
                "environment_id": environment_id,
                "created_at": created_at,
                "requested_by": requested_by,
                "description": description,
                "with_pip": bool(with_pip),
                "status": "creating",
                "ledger_event_ids": {"started": start_event["event_id"]},
            }

            builder = venv.EnvBuilder(with_pip=with_pip, clear=False, symlinks=True, upgrade=False)
            try:
                builder.create(env_path)
                duration_ms = (time.time() - start_time) * 1000.0
                python_executable = self._discover_python_executable(env_path)
                python_version = platform.python_version()
                pyvenv_cfg = env_path / "pyvenv.cfg"
                pyvenv_cfg_hash = _hash_file(pyvenv_cfg) if pyvenv_cfg.exists() else None

                success_payload = {
                    "kind": "python_vm_env_created",
                    "environment_id": environment_id,
                    "requested_by": requested_by,
                    "with_pip": bool(with_pip),
                    "status": "created",
                    "path": str(self._relative(env_path)),
                    "python_executable": str(self._relative(python_executable)),
                    "python_version": python_version,
                    "duration_ms": round(duration_ms, 3),
                }
                if description:
                    success_payload["description"] = description
                if pyvenv_cfg_hash:
                    success_payload["pyvenv_cfg_hash"] = pyvenv_cfg_hash
                success_event = self._ledger.record_event(success_payload)

                metadata.update(
                    {
                        "status": "created",
                        "python_version": python_version,
                        "duration_ms": round(duration_ms, 3),
                        "ledger_event_ids": {
                            "started": start_event["event_id"],
                            "created": success_event["event_id"],
                        },
                        "paths": {
                            "root": str(self._relative(env_path)),
                            "python_executable": str(self._relative(python_executable)),
                        },
                    }
                )
                if pyvenv_cfg_hash:
                    metadata["pyvenv_cfg_hash"] = pyvenv_cfg_hash
                metadata_path = self._write_metadata(environment_id, metadata)

                return PythonEnvironmentInfo(
                    environment_id=environment_id,
                    path=self._relative(env_path),
                    python_executable=self._relative(python_executable),
                    python_version=python_version,
                    created_at=created_at,
                    status="created",
                    with_pip=with_pip,
                    ledger_event_ids={
                        "started": start_event["event_id"],
                        "created": success_event["event_id"],
                    },
                    metadata_path=self._relative(metadata_path),
                    description=description,
                    pyvenv_cfg_hash=pyvenv_cfg_hash,
                    duration_ms=round(duration_ms, 3),
                )
            except Exception as exc:
                duration_ms = (time.time() - start_time) * 1000.0
                shutil.rmtree(env_path, ignore_errors=True)
                failure_payload = {
                    "kind": "python_vm_env_creation_failed",
                    "environment_id": environment_id,
                    "requested_by": requested_by,
                    "with_pip": bool(with_pip),
                    "status": "error",
                    "error": exc.__class__.__name__,
                    "message": str(exc),
                    "duration_ms": round(duration_ms, 3),
                }
                if description:
                    failure_payload["description"] = description
                failure_event = self._ledger.record_event(failure_payload)

                metadata.update(
                    {
                        "status": "error",
                        "error": exc.__class__.__name__,
                        "message": str(exc),
                        "duration_ms": round(duration_ms, 3),
                        "ledger_event_ids": {
                            "started": start_event["event_id"],
                            "failed": failure_event["event_id"],
                        },
                    }
                )
                metadata_path = self._write_metadata(environment_id, metadata)

                raise PythonEnvironmentError(
                    f"Environment creation failed for {environment_id}: {exc}"
                ) from exc



__all__ = [
    "InvalidPythonEnvironmentName",
    "PythonEnvironmentError",
    "PythonEnvironmentExistsError",
    "PythonEnvironmentInfo",
    "PythonEnvironmentManager",
]
