"""Artifact verification utilities for deterministic model loading."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from .model_registry import ModelRecord


class ModelArtifactVerificationError(RuntimeError):
    """Raised when a model artifact fails integrity validation."""


@dataclass
class ArtifactVerificationResult:
    """Represents the outcome of verifying a model artifact."""

    path: Path
    actual_sha256: str
    expected_sha256: Optional[str]
    size_bytes: int
    matches: bool

    def to_metadata(self) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "path": str(self.path),
            "actual_sha256": self.actual_sha256,
            "size_bytes": self.size_bytes,
            "matches": self.matches,
        }
        if self.expected_sha256 is not None:
            payload["expected_sha256"] = self.expected_sha256
        return payload


def _resolve_artifact_path(record: ModelRecord, root: Path) -> Path:
    metadata = record.metadata or {}
    download_info = metadata.get("download") if isinstance(metadata, dict) else None
    manifest_info = metadata.get("source_manifest") if isinstance(metadata, dict) else None

    path_text: Optional[str] = None
    if isinstance(download_info, dict):
        path_value = download_info.get("path")
        if isinstance(path_value, str) and path_value.strip():
            path_text = path_value.strip()

    if path_text:
        path = Path(path_text)
        if not path.is_absolute():
            path = (root / path_text).resolve()
        return path

    artifact_name: Optional[str] = None
    if isinstance(manifest_info, dict):
        artifact_value = manifest_info.get("artifact")
        if isinstance(artifact_value, str) and artifact_value.strip():
            artifact_name = artifact_value.strip()

    if artifact_name:
        return root / "cli" / "models" / record.name / artifact_name

    raise ModelArtifactVerificationError(
        f"Cannot determine artifact path for model '{record.name}'"
    )


def _expected_sha256(record: ModelRecord) -> Optional[str]:
    metadata = record.metadata or {}
    download_info = metadata.get("download") if isinstance(metadata, dict) else None
    if isinstance(download_info, dict):
        digest = download_info.get("sha256")
        if isinstance(digest, str) and digest.strip():
            return digest.strip().lower()

    manifest_info = metadata.get("source_manifest") if isinstance(metadata, dict) else None
    if isinstance(manifest_info, dict):
        digest = manifest_info.get("sha256")
        if isinstance(digest, str) and digest.strip():
            return digest.strip().lower()

    return None


def verify_model_artifact(record: ModelRecord, root: Path) -> ArtifactVerificationResult:
    """Verify the installed artifact for *record*.

    Parameters
    ----------
    record:
        The installed model record to validate.
    root:
        Repository root path used to resolve relative artifact paths.
    """

    path = _resolve_artifact_path(record, root)
    if not path.exists():
        raise ModelArtifactVerificationError(
            f"Model artifact missing for '{record.name}' at {path}"
        )

    hasher = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            hasher.update(chunk)
            size += len(chunk)
    digest = hasher.hexdigest().lower()

    expected = _expected_sha256(record)
    matches = expected is None or digest == expected

    if expected is not None and not matches:
        raise ModelArtifactVerificationError(
            "Model artifact hash mismatch: "
            f"expected {expected}, computed {digest}"
        )

    return ArtifactVerificationResult(
        path=path,
        actual_sha256=digest,
        expected_sha256=expected,
        size_bytes=size,
        matches=matches,
    )


__all__ = [
    "ArtifactVerificationResult",
    "ModelArtifactVerificationError",
    "verify_model_artifact",
]
