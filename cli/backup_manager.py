from __future__ import annotations

import hashlib
import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from cli.signature import SignatureVerifier
from cli.snapshot_ledger import SnapshotLedger


@dataclass
class BackupSummary:
    backup_dir: Path
    digest: str
    files: Sequence[Path]
    signature: str
    token_id: str


class SecureBackupManager:
    """Create signed backups of critical kernel artifacts."""

    def __init__(
        self,
        root: Path,
        ledger: SnapshotLedger,
        signature_verifier: SignatureVerifier,
    ) -> None:
        self._root = root.resolve()
        self._ledger = ledger
        self._signatures = signature_verifier
        self._backups = self._root / "cli" / "backups"
        self._backups.mkdir(parents=True, exist_ok=True)

    def _gather_files(self, paths: Sequence[Path]) -> Iterable[Path]:
        for path in paths:
            resolved = path.resolve()
            if resolved.is_file():
                yield resolved

    def create_backup(
        self,
        *,
        label: str,
        include: Sequence[Path],
        token_id: str = "default",
    ) -> BackupSummary:
        timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        backup_dir = self._backups / f"{timestamp}-{label}"
        backup_dir.mkdir(parents=True, exist_ok=True)

        copied: list[Path] = []
        digest = json.dumps([], ensure_ascii=False).encode("utf-8")
        combined = hashlib.sha256(digest)

        for file_path in self._gather_files(include):
            destination = backup_dir / file_path.name
            shutil.copy2(file_path, destination)
            copied.append(destination)
            combined.update(file_path.read_bytes())

        digest_hex = combined.hexdigest()
        signature = self._signatures.sign_digest(digest_hex, key_id=token_id)

        metadata = {
            "label": label,
            "digest": digest_hex,
            "signature": signature,
            "token_id": token_id,
            "files": [str(path.relative_to(self._root)) for path in copied],
        }
        manifest_path = backup_dir / "backup.json"
        manifest_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

        ledger_event = self._ledger.record_event(
            {
                "kind": "secure_backup_created",
                "label": label,
                "digest": digest_hex,
                "signature": signature,
                "token_id": token_id,
                "files": metadata["files"],
            }
        )

        return BackupSummary(
            backup_dir=backup_dir,
            digest=digest_hex,
            files=tuple(copied),
            signature=signature,
            token_id=token_id,
        )


__all__ = ["SecureBackupManager", "BackupSummary"]

