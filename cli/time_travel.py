"""Time travel debugging utilities for Python VM snapshots."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class TimeTravelDebugger:
    """Inspect stored snapshot state and compute diffs between checkpoints."""

    def __init__(self, workspace: Path) -> None:
        root = workspace.resolve() / "cli" / "python_vm" / "snapshots"
        self._state_dir = root / "state"
        self._session_dir = root / "sessions"

    def available_snapshots(self) -> List[int]:
        if not self._state_dir.exists():
            return []
        snapshots: List[int] = []
        for path in self._state_dir.glob("*.json"):
            try:
                snapshots.append(int(path.stem))
            except ValueError:
                continue
        return sorted(snapshots)

    def load_state(self, snapshot_id: int) -> Dict[str, object]:
        path = self._state_dir / f"{snapshot_id:08d}.json"
        if not path.exists():
            raise FileNotFoundError(path)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = {}
        if not isinstance(payload, dict):
            payload = {}
        return payload

    def load_session_metadata(self, snapshot_id: int) -> Optional[Dict[str, object]]:
        path = self._session_dir / f"pyvm-{snapshot_id:05d}.json"
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
        if isinstance(payload, dict):
            return payload
        return None

    def diff(self, from_id: int, to_id: int) -> Dict[str, object]:
        state_from = self.load_state(from_id)
        state_to = self.load_state(to_id)
        added, removed, changed = self._diff_dict(state_from, state_to)
        return {
            "from": from_id,
            "to": to_id,
            "added": added,
            "removed": removed,
            "changed": changed,
        }

    def _diff_dict(
        self,
        from_state: Dict[str, object],
        to_state: Dict[str, object],
    ) -> Tuple[Dict[str, object], Dict[str, object], Dict[str, Dict[str, object]]]:
        keys_from = set(from_state)
        keys_to = set(to_state)
        added = {k: to_state[k] for k in sorted(keys_to - keys_from)}
        removed = {k: from_state[k] for k in sorted(keys_from - keys_to)}
        changed: Dict[str, Dict[str, object]] = {}
        for key in sorted(keys_from & keys_to):
            if from_state[key] != to_state[key]:
                changed[key] = {"from": from_state[key], "to": to_state[key]}
        return added, removed, changed

