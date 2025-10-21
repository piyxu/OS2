"""Ledger diff utilities for deterministic audit comparisons."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional


def _parse_iso_timestamp(value: str) -> datetime:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return datetime.now(timezone.utc)


@dataclass(frozen=True)
class LedgerEvent:
    """Representation of snapshot ledger entries."""

    index: int
    event_id: str
    timestamp: datetime
    payload: Dict[str, object]


class LedgerAuditDiff:
    """Compare ledger entries and compute deterministic diffs."""

    def __init__(self, ledger_path: Path) -> None:
        self._ledger_path = ledger_path
        self._events = self._load_events(ledger_path)
        self._index: Dict[str, LedgerEvent] = {event.event_id: event for event in self._events}

    # ------------------------------------------------------------------
    @staticmethod
    def _load_events(path: Path) -> List[LedgerEvent]:
        events: List[LedgerEvent] = []
        if not path.exists():
            return events
        for offset, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            event_id = str(payload.get("event_id", ""))
            timestamp = _parse_iso_timestamp(str(payload.get("ts", "")))
            events.append(LedgerEvent(index=offset, event_id=event_id, timestamp=timestamp, payload=payload))
        return events

    # ------------------------------------------------------------------
    def has_event(self, event_id: str) -> bool:
        return event_id in self._index

    def get_event(self, event_id: str) -> LedgerEvent:
        try:
            return self._index[event_id]
        except KeyError as exc:
            raise KeyError(f"Event {event_id} not found in ledger") from exc

    # ------------------------------------------------------------------
    def diff(
        self,
        event_a: str,
        event_b: str,
        *,
        context: int = 5,
    ) -> Dict[str, object]:
        """Return a structured diff between two ledger entries."""

        left = self.get_event(event_a)
        right = self.get_event(event_b)
        context = max(int(context), 0)

        payload_diff = self._diff_payload(left.payload, right.payload)
        context_a = self._context_window(left.index, context)
        context_b = self._context_window(right.index, context)
        delta_ms = int((right.timestamp - left.timestamp).total_seconds() * 1000)

        return {
            "left": self._event_metadata(left),
            "right": self._event_metadata(right),
            "timestamp_delta_ms": delta_ms,
            "payload_diff": payload_diff,
            "context": {
                "left": context_a,
                "right": context_b,
            },
        }

    # ------------------------------------------------------------------
    def _context_window(self, index: int, length: int) -> List[Dict[str, object]]:
        if length <= 0:
            return []
        start = max(index - length, 0)
        window = self._events[start:index]
        return [self._event_metadata(event) for event in window]

    @staticmethod
    def _event_metadata(event: LedgerEvent) -> Dict[str, object]:
        payload = dict(event.payload)
        payload.pop("signature", None)
        return {
            "index": event.index,
            "event_id": event.event_id,
            "timestamp": event.timestamp.isoformat(),
            "kind": payload.get("kind"),
            "payload": payload,
        }

    @staticmethod
    def _diff_payload(
        left: Dict[str, object],
        right: Dict[str, object],
    ) -> Dict[str, object]:
        keys_left = set(left.keys())
        keys_right = set(right.keys())

        only_left = {key: left[key] for key in sorted(keys_left - keys_right)}
        only_right = {key: right[key] for key in sorted(keys_right - keys_left)}

        changed: Dict[str, Dict[str, object]] = {}
        for key in sorted(keys_left & keys_right):
            if left[key] != right[key]:
                changed[key] = {"left": left[key], "right": right[key]}

        return {
            "only_left": only_left,
            "only_right": only_right,
            "changed": changed,
        }


__all__ = ["LedgerAuditDiff", "LedgerEvent"]
