from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List

from cli.ai_replay import AIReplayManager
from cli.snapshot_ledger import SnapshotLedger


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.isoformat().replace("+00:00", "Z")


@dataclass
class SecurityEvent:
    event_id: str
    event_type: str
    message: str
    severity: str
    recorded_at: str


class SecurityLogManager:
    """Persist security events and optionally integrate them with replay storage."""

    def __init__(self, root: Path, ledger: SnapshotLedger) -> None:
        self._root = root.resolve()
        self._ledger = ledger
        self._lock = threading.RLock()
        self._log_path = self._root / "cli" / "data" / "security_log.jsonl"
        self._log_path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, *, event_type: str, message: str, severity: str = "info") -> SecurityEvent:
        payload = {
            "kind": "security_event_recorded",
            "event_type": event_type,
            "message": message,
            "severity": severity,
            "recorded_at": _iso(_now()),
        }
        ledger_event = self._ledger.record_event(payload)
        entry = {
            "event_id": ledger_event.get("event_id", ""),
            "event_type": event_type,
            "message": message,
            "severity": severity,
            "recorded_at": payload["recorded_at"],
        }
        with self._lock:
            with self._log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return SecurityEvent(**entry)

    def iter_events(self) -> Iterable[SecurityEvent]:
        if not self._log_path.exists():
            return
        with self._log_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(payload, dict):
                    continue
                yield SecurityEvent(
                    event_id=str(payload.get("event_id", "")),
                    event_type=str(payload.get("event_type", "")),
                    message=str(payload.get("message", "")),
                    severity=str(payload.get("severity", "info")),
                    recorded_at=str(payload.get("recorded_at", "")),
                )

    def integrate_with_replay(self, replay: AIReplayManager) -> Dict[str, List[str]]:
        integrated: List[str] = []
        for event in self.iter_events():
            payload = {
                "event_type": event.event_type,
                "message": event.message,
                "severity": event.severity,
                "recorded_at": event.recorded_at,
                "event_id": event.event_id,
            }
            path = replay.integrate_security_event(payload)
            integrated.append(str(path))
        self._ledger.record_event(
            {
                "kind": "security_log_integrated",
                "records": integrated,
                "count": len(integrated),
            }
        )
        return {"records": integrated, "count": len(integrated)}


__all__ = ["SecurityLogManager", "SecurityEvent"]

