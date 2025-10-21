"""Synchronize Python ``sys.path`` entries with the snapshot ledger."""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from cli.snapshot_ledger import SnapshotLedger


class PythonSysPathRegistry:
    """Persist interpreter ``sys.path`` state and emit ledger events."""

    def __init__(self, root: Path, ledger: SnapshotLedger) -> None:
        self._workspace = root.resolve()
        self._ledger = ledger
        self._lock = threading.RLock()
        self._root = self._workspace / "cli" / "python_vm" / "syspaths"
        self._root.mkdir(parents=True, exist_ok=True)
        self._sessions_dir = self._root / "sessions"
        self._sessions_dir.mkdir(parents=True, exist_ok=True)
        self._state_path = self._root / "state.json"
        self._state: Dict[str, object] = self._load_state()

    # ------------------------------------------------------------------
    def _load_state(self) -> Dict[str, object]:
        try:
            payload = json.loads(self._state_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {"paths": [], "event_id": None, "hash": None}
        except json.JSONDecodeError:
            return {"paths": [], "event_id": None, "hash": None}
        paths = payload.get("paths", [])
        event_id = payload.get("event_id")
        hash_value = payload.get("hash")
        return {"paths": list(paths), "event_id": event_id, "hash": hash_value}

    def _write_state(self) -> None:
        with self._state_path.open("w", encoding="utf-8") as handle:
            json.dump(self._state, handle, ensure_ascii=False, indent=2)
            handle.write("\n")

    # ------------------------------------------------------------------
    def _serialize_paths(self, paths: Sequence[str]) -> List[Dict[str, object]]:
        entries: List[Dict[str, object]] = []
        for index, raw in enumerate(paths):
            raw_path = raw or ""
            resolved = str((Path(raw_path) if raw_path else self._workspace).resolve())
            within_workspace = resolved.startswith(str(self._workspace))
            try:
                relative = (Path(resolved).relative_to(self._workspace).as_posix()
                            if within_workspace
                            else None)
            except ValueError:  # pragma: no cover - defensive
                relative = None
            entries.append(
                {
                    "index": index,
                    "raw": raw_path,
                    "resolved": resolved,
                    "relative": relative,
                    "within_workspace": within_workspace,
                }
            )
        return entries

    def _hash_paths(self, paths: Iterable[str]) -> str:
        serialized = json.dumps(list(paths), ensure_ascii=False)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------
    def synchronize(
        self,
        *,
        session_id: str,
        sandbox_id: str,
        sys_path: Sequence[str],
        injected_paths: Sequence[str],
        snapshot_id: int,
    ) -> Dict[str, object]:
        """Record *sys_path* in the ledger and persist local state."""

        with self._lock:
            serialized = self._serialize_paths(sys_path)
            current_hash = self._hash_paths(entry["raw"] for entry in serialized)
            previous_hash = self._state.get("hash")
            changed = current_hash != previous_hash
            event = self._ledger.record_event(
                {
                    "kind": "python_vm_syspath_synced",
                    "session_id": session_id,
                    "sandbox_id": sandbox_id,
                    "paths": serialized,
                    "count": len(serialized),
                    "injected": list(injected_paths),
                    "paths_hash": current_hash,
                    "previous_hash": previous_hash,
                    "changed": changed,
                    "previous_event_id": self._state.get("event_id"),
                    "snapshot_id": int(snapshot_id),
                }
            )
            self._state = {
                "paths": [entry["raw"] for entry in serialized],
                "event_id": event.get("event_id"),
                "hash": current_hash,
            }
            self._write_state()

            session_payload = {
                "session_id": session_id,
                "sandbox_id": sandbox_id,
                "event_id": event.get("event_id"),
                "paths": serialized,
                "injected": list(injected_paths),
                "paths_hash": current_hash,
                "snapshot_id": int(snapshot_id),
            }
            session_path = self._sessions_dir / f"{session_id}.json"
            with session_path.open("w", encoding="utf-8") as handle:
                json.dump(session_payload, handle, ensure_ascii=False, indent=2)
                handle.write("\n")

            return event


__all__ = ["PythonSysPathRegistry"]
