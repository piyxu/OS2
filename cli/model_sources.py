"""Utilities for loading model source manifests."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional


class ModelSourceError(RuntimeError):
    """Raised when a manifest cannot be parsed."""


@dataclass
class ModelSource:
    """Represents a model source defined by a JSON manifest."""

    name: str
    provider: str
    url: str
    sha256: str
    token_cost: int
    artifact: str
    capability: Optional[str]
    metadata: Dict[str, object]
    signature: Optional[str]
    signature_key: Optional[str]
    signature_algorithm: str = "hmac-sha256"

    def to_metadata(self) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "manifest_url": self.url,
            "sha256": self.sha256,
            "artifact": self.artifact,
            "token_cost": self.token_cost,
            "provider": self.provider,
            "metadata": dict(self.metadata),
        }
        if self.signature:
            payload["signature"] = self.signature
            payload["signature_algorithm"] = self.signature_algorithm
        if self.signature_key:
            payload["signature_key"] = self.signature_key
        return payload


def load_manifest(path: Path, *, default_name: str, default_provider: str) -> ModelSource:
    """Load a manifest from *path* and convert it into a :class:`ModelSource`."""

    if not path.exists():
        raise ModelSourceError(f"Manifest not found: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive
        raise ModelSourceError(f"Invalid JSON manifest: {exc}") from exc

    if not isinstance(raw, dict):
        raise ModelSourceError("Manifest must be a JSON object")

    url = raw.get("url")
    if not url or not isinstance(url, str):
        raise ModelSourceError("Manifest missing required 'url' field")
    if not url.startswith("https://"):
        raise ModelSourceError("Manifest url must start with 'https://'")

    sha256 = raw.get("sha256")
    if not sha256 or not isinstance(sha256, str):
        raise ModelSourceError("Manifest missing required 'sha256' field")

    token_cost = raw.get("token_cost", 0)
    if not isinstance(token_cost, int) or token_cost < 0:
        raise ModelSourceError("Manifest 'token_cost' must be a non-negative integer")

    artifact = raw.get("artifact") or Path(url).name or f"{default_name}.bin"
    capability = raw.get("capability")
    provider = str(raw.get("provider", default_provider))

    metadata: Dict[str, object] = {}
    extra_meta = raw.get("metadata", {})
    if isinstance(extra_meta, dict):
        metadata.update(extra_meta)

    signature = raw.get("signature")
    if signature is not None and not isinstance(signature, str):
        raise ModelSourceError("Manifest 'signature' field must be a string when present")

    signature_key = raw.get("signature_key")
    if signature_key is not None and not isinstance(signature_key, str):
        raise ModelSourceError("Manifest 'signature_key' must be a string when present")

    signature_algorithm = raw.get("signature_algorithm", "hmac-sha256")
    if not isinstance(signature_algorithm, str):
        raise ModelSourceError("Manifest 'signature_algorithm' must be a string")

    name = str(raw.get("name", default_name))

    return ModelSource(
        name=name,
        provider=provider,
        url=url,
        sha256=sha256,
        token_cost=token_cost,
        artifact=str(artifact),
        capability=capability if isinstance(capability, str) else None,
        metadata=metadata,
        signature=signature,
        signature_key=signature_key,
        signature_algorithm=signature_algorithm,
    )


__all__ = ["ModelSource", "ModelSourceError", "load_manifest"]
