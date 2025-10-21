from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from cli.snapshot_ledger import SnapshotLedger


@dataclass
class EntropyDeviation:
    event_id: str
    entropy_bits: int
    expected_multiple: int
    kind: str


class EntropyAuditor:
    """Review entropy events from the snapshot ledger for irregularities."""

    def __init__(self, root: Path, ledger: SnapshotLedger) -> None:
        self._root = root.resolve()
        self._ledger = ledger
        self._ledger_path = self._root / "cli" / "data" / "snapshot_ledger.jsonl"

    def audit(self, *, limit: Optional[int] = None) -> Dict[str, object]:
        deviations: List[EntropyDeviation] = []
        total_events = 0
        if not self._ledger_path.exists():
            return {"total_events": 0, "deviations": []}

        with self._ledger_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(payload, dict):
                    continue
                if "entropy_bits" not in payload:
                    continue
                total_events += 1
                entropy_bits = int(payload.get("entropy_bits", 0))
                if entropy_bits % 8 != 0 or entropy_bits <= 0:
                    deviations.append(
                        EntropyDeviation(
                            event_id=str(payload.get("event_id", "")),
                            entropy_bits=entropy_bits,
                            expected_multiple=8,
                            kind=str(payload.get("kind", "")),
                        )
                    )
                if limit is not None and len(deviations) >= limit:
                    break

        result = {
            "total_events": total_events,
            "deviations": [deviation.__dict__ for deviation in deviations],
        }

        if deviations:
            self._ledger.record_event(
                {
                    "kind": "entropy_audit_deviation_detected",
                    "total_events": total_events,
                    "deviations": result["deviations"],
                }
            )
        else:
            self._ledger.record_event(
                {
                    "kind": "entropy_audit_completed",
                    "total_events": total_events,
                }
            )

        return result


__all__ = ["EntropyAuditor", "EntropyDeviation"]

