"""Content-addressable storage for downloaded model artifacts."""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path


class ContentAddressableStoreError(RuntimeError):
    """Raised when storing an artifact in the CAS fails."""


class ContentAddressableStore:
    """Persist files under their content hash."""

    def __init__(self, root: Path, *, algorithm: str = "sha256") -> None:
        self._root = root
        self._root.mkdir(parents=True, exist_ok=True)
        self._algorithm = algorithm

    def store(self, path: Path, digest: str) -> Path:
        """Store *path* in the CAS and return the canonical location."""

        source = path.resolve()
        if not source.exists():
            raise ContentAddressableStoreError(f"Source file does not exist: {source}")

        try:
            hasher = hashlib.new(self._algorithm)
        except ValueError as exc:  # pragma: no cover - defensive
            raise ContentAddressableStoreError(str(exc)) from exc

        with source.open("rb") as handle:
            for chunk in iter(lambda: handle.read(65536), b""):
                hasher.update(chunk)

        computed = hasher.hexdigest()
        if computed.lower() != digest.lower():
            raise ContentAddressableStoreError(
                "Digest mismatch while storing artifact in CAS"
            )

        subdir = self._root / computed[:2]
        subdir.mkdir(parents=True, exist_ok=True)
        target = subdir / computed
        if target.exists():
            return target

        temp_path = target.with_suffix(".tmp")
        shutil.copy2(source, temp_path)
        temp_path.replace(target)
        return target


__all__ = ["ContentAddressableStore", "ContentAddressableStoreError"]
