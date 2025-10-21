"""Task proposal registry for Roken Assembly integrations."""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from .snapshot_ledger import SnapshotLedger


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _isoformat(dt: datetime) -> str:
    return dt.isoformat().replace("+00:00", "Z")


ALLOWED_SOURCES = {
    "roken-assembly": "Roken Assembly evolution engine",
    "human-operator": "Human operator",  # fallback path for manual entries
}


class TaskProposalError(RuntimeError):
    """Raised when task proposal operations cannot be completed."""


@dataclass
class TaskProposal:
    """Represents an individual task proposal entry."""

    proposal_id: int
    source: str
    title: str
    description: str = ""
    tags: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: _isoformat(_now_utc()))

    def to_dict(self) -> Dict[str, object]:
        return {
            "proposal_id": self.proposal_id,
            "source": self.source,
            "title": self.title,
            "description": self.description,
            "tags": list(self.tags),
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "TaskProposal":
        proposal_id = int(payload.get("proposal_id", 0))
        source = str(payload.get("source", ""))
        title = str(payload.get("title", ""))
        description = str(payload.get("description", ""))
        created_at = str(payload.get("created_at", ""))
        raw_tags = payload.get("tags", [])
        tags: List[str] = []
        if isinstance(raw_tags, list):
            tags = [str(tag) for tag in raw_tags]
        proposal = cls(
            proposal_id=proposal_id,
            source=source,
            title=title,
            description=description,
            tags=tags,
        )
        if created_at:
            proposal.created_at = created_at
        return proposal


class TaskProposalRegistry:
    """Persistent registry storing task proposals."""

    def __init__(self, root: Path, ledger: SnapshotLedger) -> None:
        self._ledger = ledger
        self._path = root / "cli" / "data" / "task_proposals.json"
        self._lock = threading.RLock()
        self._proposals = self._load_or_initialize()

    def list(self) -> List[Dict[str, object]]:
        with self._lock:
            return [proposal.to_dict() for proposal in self._proposals]

    def register(
        self,
        source: str,
        title: str,
        *,
        description: str = "",
        tags: Optional[List[str]] = None,
    ) -> Dict[str, object]:
        normalized = source.strip().lower()
        if normalized not in ALLOWED_SOURCES:
            raise TaskProposalError(f"Unknown proposal source: {source}")
        clean_tags = []
        if tags:
            clean_tags = sorted({tag.strip().lower() for tag in tags if tag.strip()})

        with self._lock:
            proposal_id = self._next_id_locked()
            proposal = TaskProposal(
                proposal_id=proposal_id,
                source=normalized,
                title=title.strip(),
                description=description.strip(),
                tags=clean_tags,
            )
            self._proposals.append(proposal)
            self._save_locked()

        ledger_event = self._ledger.record_event(
            {
                "kind": "task_proposal_registered",
                "source": normalized,
                "proposal": proposal.to_dict(),
            }
        )
        return {"proposal": proposal.to_dict(), "ledger_event": ledger_event}

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------
    def _load_or_initialize(self) -> List[TaskProposal]:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if not self._path.exists():
            self._path.write_text("[]\n", encoding="utf-8")
            return []
        try:
            payload = json.loads(self._path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = []
        proposals: List[TaskProposal] = []
        if isinstance(payload, list):
            for item in payload:
                if isinstance(item, dict):
                    proposals.append(TaskProposal.from_dict(item))
        return proposals

    def _save_locked(self) -> None:
        body = [proposal.to_dict() for proposal in self._proposals]
        self._path.write_text(json.dumps(body, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def _next_id_locked(self) -> int:
        if not self._proposals:
            return 1
        return max(proposal.proposal_id for proposal in self._proposals) + 1


__all__ = [
    "TaskProposalRegistry",
    "TaskProposalError",
    "TaskProposal",
    "ALLOWED_SOURCES",
]

