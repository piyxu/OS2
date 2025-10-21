"""Snapshot ledger inspection utilities."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional


@dataclass
class LedgerEntry:
    index: int
    event_id: str
    kind: str
    timestamp: Optional[str]
    payload: Dict[str, object]

    def compact(self) -> Dict[str, object]:
        return {
            "index": self.index,
            "event_id": self.event_id,
            "kind": self.kind,
            "timestamp": self.timestamp,
            "payload": self.payload,
        }


class LedgerInspector:
    """Inspect and summarise snapshot ledger entries."""

    def __init__(self, ledger_path: Path) -> None:
        self._ledger_path = ledger_path
        self._entries = self._load_entries()

    def _load_entries(self) -> List[LedgerEntry]:
        if not self._ledger_path.exists():
            return []
        entries: List[LedgerEntry] = []
        for index, line in enumerate(self._ledger_path.read_text(encoding="utf-8").splitlines()):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                continue
            event_id = str(payload.get("event_id", ""))
            kind = str(payload.get("kind", "unknown"))
            timestamp = payload.get("ts")
            if timestamp is not None:
                timestamp = str(timestamp)
            entries.append(
                LedgerEntry(
                    index=index,
                    event_id=event_id,
                    kind=kind,
                    timestamp=timestamp,
                    payload=payload,
                )
            )
        return entries

    @property
    def total(self) -> int:
        return len(self._entries)

    def summary(self) -> Dict[str, object]:
        counter = Counter(entry.kind for entry in self._entries)
        return {
            "total_events": self.total,
            "by_kind": dict(counter),
        }

    def tail(self, limit: Optional[int] = None, kind: Optional[str] = None) -> List[Dict[str, object]]:
        entries: Iterable[LedgerEntry] = self._entries
        if kind:
            entries = [entry for entry in entries if entry.kind == kind]
        if limit is not None and limit > 0:
            entries = list(entries)[-limit:]
        else:
            entries = list(entries)
        return [entry.compact() for entry in entries]


__all__ = ["LedgerInspector", "LedgerEntry"]
