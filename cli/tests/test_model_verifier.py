from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from cli.model_registry import ModelRecord
from cli.model_verifier import (
    ModelArtifactVerificationError,
    verify_model_artifact,
)


def _record_for(artifact_path: Path, digest: str) -> ModelRecord:
    return ModelRecord(
        name="demo",
        source="https://example.com/demo",
        provider="local",
        manifest=None,
        installed_at="2024-01-01T00:00:00Z",
        metadata={
            "download": {
                "path": str(artifact_path),
                "sha256": digest,
            },
            "source_manifest": {
                "sha256": digest,
                "artifact": artifact_path.name,
            },
        },
        capability="cap.model.demo",
    )


def test_verify_model_artifact_success(tmp_path: Path) -> None:
    artifact = tmp_path / "weights.bin"
    payload = b"deterministic weights"
    artifact.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    record = _record_for(artifact, digest)

    result = verify_model_artifact(record, tmp_path)

    assert result.matches is True
    assert result.actual_sha256 == digest
    assert result.expected_sha256 == digest
    assert result.size_bytes == len(payload)


def test_verify_model_artifact_mismatch(tmp_path: Path) -> None:
    artifact = tmp_path / "weights.bin"
    artifact.write_bytes(b"content")
    digest = hashlib.sha256(b"different").hexdigest()
    record = _record_for(artifact, digest)

    with pytest.raises(ModelArtifactVerificationError):
        verify_model_artifact(record, tmp_path)


def test_verify_model_missing_file(tmp_path: Path) -> None:
    artifact = tmp_path / "weights.bin"
    digest = hashlib.sha256(b"data").hexdigest()
    record = _record_for(artifact, digest)

    with pytest.raises(ModelArtifactVerificationError):
        verify_model_artifact(record, tmp_path)
