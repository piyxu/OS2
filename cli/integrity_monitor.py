from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from cli.snapshot_ledger import SnapshotLedger


@dataclass
class IntegrityReport:
    label: str
    digest: str
    files_hashed: int


class IntegrityMonitor:
    """Compute deterministic hashes for critical files and record ledger events."""

    def __init__(self, root: Path, ledger: SnapshotLedger) -> None:
        self._root = root.resolve()
        self._ledger = ledger

    def _iter_files(self, paths: Sequence[Path]) -> Iterable[Path]:
        for path in paths:
            resolved = path.resolve()
            if resolved.is_file():
                yield resolved
            elif resolved.is_dir():
                for nested in sorted(resolved.rglob("*")):
                    if nested.is_file():
                        yield nested

    def run_check(self, *, label: str, paths: Sequence[Path]) -> IntegrityReport:
        digest = hashlib.sha256()
        count = 0
        for file_path in self._iter_files(paths):
            count += 1
            digest.update(str(file_path.relative_to(self._root)).encode("utf-8"))
            digest.update(file_path.read_bytes())
        summary = IntegrityReport(label=label, digest=digest.hexdigest(), files_hashed=count)
        self._ledger.record_event(
            {
                "kind": "integrity_monitor_run",
                "label": summary.label,
                "digest": summary.digest,
                "files_hashed": summary.files_hashed,
            }
        )
        return summary


__all__ = ["IntegrityMonitor", "IntegrityReport"]

