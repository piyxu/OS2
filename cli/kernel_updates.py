"""Kernel update distribution with token-signed package verification."""

from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from .signature import SignatureError, SignatureVerifier
from .snapshot_ledger import SnapshotLedger


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _isoformat(dt: datetime) -> str:
    return dt.isoformat().replace("+00:00", "Z")


class KernelUpdateError(RuntimeError):
    """Raised when kernel update distribution cannot proceed."""


@dataclass
class KernelUpdatePackage:
    """Metadata recorded for a distributed kernel update package."""

    package_id: int
    version: str
    artifact: str
    sha256: str
    size_bytes: int
    token_id: str
    signature: str
    signature_algorithm: str = "hmac-sha256"
    notes: str = ""
    distributed_at: str = field(default_factory=lambda: _isoformat(_now_utc()))

    def to_dict(self) -> Dict[str, object]:
        return {
            "package_id": self.package_id,
            "version": self.version,
            "artifact": self.artifact,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "token_id": self.token_id,
            "signature": self.signature,
            "signature_algorithm": self.signature_algorithm,
            "notes": self.notes,
            "distributed_at": self.distributed_at,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "KernelUpdatePackage":
        package = cls(
            package_id=int(payload.get("package_id", 0)),
            version=str(payload.get("version", "")),
            artifact=str(payload.get("artifact", "")),
            sha256=str(payload.get("sha256", "")),
            size_bytes=int(payload.get("size_bytes", 0)),
            token_id=str(payload.get("token_id", "")),
            signature=str(payload.get("signature", "")),
            signature_algorithm=str(payload.get("signature_algorithm", "hmac-sha256")),
            notes=str(payload.get("notes", "")),
        )
        distributed_at = payload.get("distributed_at")
        if distributed_at:
            package.distributed_at = str(distributed_at)
        return package


class KernelUpdateDistributor:
    """Manage token-signed kernel update distribution events."""

    def __init__(
        self,
        root: Path,
        ledger: SnapshotLedger,
        signature_verifier: SignatureVerifier,
    ) -> None:
        self._root = root.resolve()
        self._ledger = ledger
        self._signature_verifier = signature_verifier
        self._path = self._root / "cli" / "data" / "kernel_updates.json"
        self._lock = threading.RLock()
        self._packages = self._load_or_initialize()

    def list_packages(self) -> List[Dict[str, object]]:
        with self._lock:
            return [package.to_dict() for package in self._packages]

    def distribute(
        self,
        version: str,
        artifact_path: Path,
        *,
        sha256: str,
        token_id: str,
        signature: str,
        signature_algorithm: str = "hmac-sha256",
        notes: str = "",
    ) -> Dict[str, object]:
        resolved = artifact_path.resolve()
        try:
            relative_path = resolved.relative_to(self._root)
        except ValueError as exc:
            raise KernelUpdateError("Artifact path must be inside the workspace root") from exc
        if not resolved.exists() or not resolved.is_file():
            raise KernelUpdateError(f"Kernel update artifact not found: {resolved}")

        computed = self._hash_file(resolved)
        if computed.lower() != sha256.lower():
            raise KernelUpdateError("SHA256 mismatch for kernel update artifact")

        if signature_algorithm.lower() != "hmac-sha256":
            raise KernelUpdateError(
                f"Unsupported signature algorithm: {signature_algorithm}"
            )

        try:
            self._signature_verifier.verify_digest(
                computed,
                signature,
                key_id=token_id or None,
                algorithm=signature_algorithm,
            )
        except SignatureError as exc:
            raise KernelUpdateError(f"Signature verification failed: {exc}") from exc

        size_bytes = resolved.stat().st_size

        with self._lock:
            package_id = self._next_id_locked()
            package = KernelUpdatePackage(
                package_id=package_id,
                version=version.strip(),
                artifact=str(relative_path),
                sha256=computed,
                size_bytes=size_bytes,
                token_id=token_id or "default",
                signature=signature.strip().lower(),
                signature_algorithm=signature_algorithm.strip().lower(),
                notes=notes.strip(),
            )
            self._packages.append(package)
            self._save_locked()

        ledger_event = self._ledger.record_event(
            {
                "kind": "kernel_update_distributed",
                "package": package.to_dict(),
            }
        )
        return {"package": package.to_dict(), "ledger_event": ledger_event}

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------
    def _load_or_initialize(self) -> List[KernelUpdatePackage]:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if not self._path.exists():
            self._path.write_text("[]\n", encoding="utf-8")
            return []
        try:
            payload = json.loads(self._path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = []
        packages: List[KernelUpdatePackage] = []
        if isinstance(payload, list):
            for item in payload:
                if isinstance(item, dict):
                    packages.append(KernelUpdatePackage.from_dict(item))
        return packages

    def _save_locked(self) -> None:
        body = [package.to_dict() for package in self._packages]
        self._path.write_text(
            json.dumps(body, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    def _next_id_locked(self) -> int:
        if not self._packages:
            return 1
        return max(package.package_id for package in self._packages) + 1

    @staticmethod
    def _hash_file(path: Path) -> str:
        hasher = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(65536), b""):
                hasher.update(chunk)
        return hasher.hexdigest()


__all__ = [
    "KernelUpdateDistributor",
    "KernelUpdateError",
    "KernelUpdatePackage",
]
