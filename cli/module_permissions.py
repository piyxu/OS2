from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Dict, List, Sequence, Set

from cli.snapshot_ledger import SnapshotLedger


class ModulePermissionError(RuntimeError):
    """Raised when module permissions cannot be updated."""


class ModulePermissionRegistry:
    """Persist and audit module permissions bound to capability tokens."""

    def __init__(self, root: Path, ledger: SnapshotLedger) -> None:
        self._root = root.resolve()
        self._ledger = ledger
        self._lock = threading.RLock()
        self._path = self._root / "cli" / "data" / "module_permissions.json"
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._state: Dict[str, List[str]] = self._load()

    # ------------------------------------------------------------------
    def _load(self) -> Dict[str, List[str]]:
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {"default": ["__all__"]}
        except json.JSONDecodeError:
            return {"default": ["__all__"]}
        if not isinstance(data, dict):  # pragma: no cover - defensive
            return {"default": ["__all__"]}
        normalized: Dict[str, List[str]] = {}
        for token_id, modules in data.items():
            if isinstance(modules, list):
                normalized[token_id] = [str(module) for module in modules if module]
        normalized.setdefault("default", ["__all__"])
        return normalized

    def _write(self) -> None:
        with self._path.open("w", encoding="utf-8") as handle:
            json.dump(self._state, handle, ensure_ascii=False, indent=2)
            handle.write("\n")

    # ------------------------------------------------------------------
    def list_permissions(self) -> Dict[str, List[str]]:
        with self._lock:
            return {token: list(modules) for token, modules in self._state.items()}

    # ------------------------------------------------------------------
    def grant(self, token_id: str, module: str) -> Dict[str, object]:
        token = token_id or "default"
        pattern = module.strip()
        if not pattern:
            raise ModulePermissionError("Module pattern must not be empty")

        with self._lock:
            modules = self._state.setdefault(token, [])
            if pattern in modules:
                return {"token_id": token, "module": pattern, "ledger_event": None}
            modules.append(pattern)
            self._write()

        ledger_event = self._ledger.record_event(
            {
                "kind": "python_module_permission_granted",
                "token_id": token,
                "module": pattern,
            }
        )
        return {"token_id": token, "module": pattern, "ledger_event": ledger_event}

    # ------------------------------------------------------------------
    def revoke(self, token_id: str, module: str) -> Dict[str, object]:
        token = token_id or "default"
        pattern = module.strip()
        with self._lock:
            modules = self._state.get(token, [])
            if pattern not in modules:
                raise ModulePermissionError("Module pattern not present for token")
            modules.remove(pattern)
            if not modules and token != "default":
                self._state.pop(token, None)
            self._write()

        ledger_event = self._ledger.record_event(
            {
                "kind": "python_module_permission_revoked",
                "token_id": token,
                "module": pattern,
            }
        )
        return {"token_id": token, "module": pattern, "ledger_event": ledger_event}

    # ------------------------------------------------------------------
    def is_allowed(self, module: str, tokens: Sequence[str]) -> bool:
        allowed = self.allowed_modules(tokens)
        if "__all__" in allowed:
            return True
        for pattern in allowed:
            if pattern.endswith(".*"):
                prefix = pattern[:-2]
                if module == prefix or module.startswith(prefix + "."):
                    return True
            elif pattern == module:
                return True
        return False

    def allowed_modules(self, tokens: Sequence[str]) -> Set[str]:
        token_set = [token for token in tokens if token]
        if not token_set:
            token_set = ["default"]

        with self._lock:
            granted: Set[str] = set()
            for token in token_set:
                granted.update(self._state.get(token, []))
            if "default" not in token_set:
                granted.update(self._state.get("default", []))
        return granted


__all__ = [
    "ModulePermissionRegistry",
    "ModulePermissionError",
]

