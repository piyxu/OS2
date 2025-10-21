"""Deterministic import verification for the embedded Python VM."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import sysconfig
import threading
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Callable, Dict, Iterable, Optional, Sequence

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from cli.snapshot_ledger import SnapshotLedger
from cli.token_sandbox import TokenSandbox
from cli.module_permissions import ModulePermissionRegistry


def compute_file_hash(path: Path) -> str:
    """Return the SHA-256 hash for *path* using a streaming reader."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


class ImportVerificationError(ImportError):
    """Raised when a module import fails verification."""


@dataclass
class _ImportDecision:
    """Intermediate metadata describing how an import should be handled."""

    module: str
    origin: Optional[Path]
    status: str
    manifest_entry: Optional[Dict[str, str]]
    computed_hash: Optional[str]
    signing_key: Optional[str] = None
    signature_validated: bool = False


class DeterministicImportVerifier:
    """Validate module hashes before allowing Python VM imports."""

    def __init__(
        self,
        root: Path,
        ledger: SnapshotLedger,
        *,
        manifest_path: Optional[Path] = None,
    ) -> None:
        self._root = root.resolve()
        self._ledger = ledger
        self._lock = threading.RLock()
        sessions_dir = self._root / "cli" / "python_vm"
        sessions_dir.mkdir(parents=True, exist_ok=True)
        self._manifest_path = (
            manifest_path
            if manifest_path is not None
            else sessions_dir / "import_manifest.json"
        )
        self._keys_path = sessions_dir / "import_signers.json"
        self._manifest: Dict[str, Dict[str, str]] = self._load_manifest()
        self._signing_keys: Dict[str, str] = self._load_signing_keys()
        self._stdlib_paths = self._discover_stdlib_paths()
        self._safe_deny_prefixes: Sequence[str] = (
            "socket",
            "ssl",
            "http",
            "urllib",
            "requests",
            "ftplib",
            "wsgiref",
            "asyncio",
        )

    # ------------------------------------------------------------------
    def _load_manifest(self) -> Dict[str, Dict[str, str]]:
        try:
            raw = json.loads(self._manifest_path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):  # pragma: no cover - defensive
                return {}
            normalized: Dict[str, Dict[str, str]] = {}
            for key, value in raw.items():
                if isinstance(value, dict):
                    normalized[key] = {
                        "path": str(value.get("path", "")),
                        "hash": str(value.get("hash", "")),
                    }
            return normalized
        except FileNotFoundError:
            self._manifest_path.parent.mkdir(parents=True, exist_ok=True)
            self._manifest_path.write_text("{}\n", encoding="utf-8")
            return {}
        except json.JSONDecodeError:  # pragma: no cover - defensive
            return {}

    def _discover_stdlib_paths(self) -> Sequence[Path]:
        paths: Iterable[str] = sysconfig.get_paths().values()
        resolved = []
        for item in paths:
            if not item:
                continue
            resolved.append(Path(item).resolve())
        return tuple(resolved)

    def _load_signing_keys(self) -> Dict[str, str]:
        try:
            raw = json.loads(self._keys_path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                return {str(key): str(value) for key, value in raw.items()}
        except FileNotFoundError:
            self._keys_path.parent.mkdir(parents=True, exist_ok=True)
            self._keys_path.write_text("{}\n", encoding="utf-8")
        except json.JSONDecodeError:
            return {}
        return {}

    # ------------------------------------------------------------------
    def create_import_hook(
        self,
        *,
        session_id: str,
        sandbox: TokenSandbox,
        default_import: Callable[..., ModuleType],
        permission_tokens: Optional[Sequence[str]] = None,
        permission_manager: Optional[ModulePermissionRegistry] = None,
    ) -> Callable[..., ModuleType]:
        """Return a custom ``__import__`` hook bound to *session_id*."""

        tokens = tuple(permission_tokens or ())

        def _import_hook(
            name: str,
            globals_dict: Optional[Dict[str, object]] = None,
            locals_dict: Optional[Dict[str, object]] = None,
            fromlist: Optional[Sequence[str]] = None,
            level: int = 0,
        ) -> ModuleType:
            resolved_name = self._resolve_name(name, globals_dict, level)
            decision = self._prepare_import(
                resolved_name,
                session_id,
                sandbox,
                permission_tokens=tokens,
                permission_manager=permission_manager,
            )
            try:
                module = default_import(
                    name, globals_dict, locals_dict, fromlist or (), level
                )
            except Exception as exc:
                self._record_event(decision, session_id, sandbox, outcome="error", error=str(exc))
                raise
            else:
                self._record_event(decision, session_id, sandbox, outcome="ok")
                return module

        return _import_hook

    # ------------------------------------------------------------------
    def _resolve_name(
        self,
        name: str,
        globals_dict: Optional[Dict[str, object]],
        level: int,
    ) -> str:
        if level <= 0:
            return name
        package = None
        if globals_dict:
            package = globals_dict.get("__package__") or globals_dict.get("__name__")
        return importlib.util.resolve_name(name, package or "__main__")

    # ------------------------------------------------------------------
    def _prepare_import(
        self,
        module: str,
        session_id: str,
        sandbox: TokenSandbox,
        *,
        permission_tokens: Sequence[str],
        permission_manager: Optional[ModulePermissionRegistry],
    ) -> _ImportDecision:
        spec = importlib.util.find_spec(module)
        if spec is None:
            message = f"Module {module!r} could not be found"
            self._record_blocked(
                module,
                session_id,
                sandbox,
                reason="missing_spec",
                details=message,
                tokens=list(permission_tokens),
            )
            raise ImportVerificationError(message)

        safe_mode_active = bool(sandbox.metadata.get("safe_mode"))

        origin = Path(spec.origin).resolve() if spec.origin and spec.origin not in {"built-in", "frozen"} else None

        if safe_mode_active and self._is_network_module(module):
            message = f"Module {module!r} is blocked in safe mode"
            self._record_blocked(
                module,
                session_id,
                sandbox,
                reason="safe_mode_network_blocked",
                details=message,
                origin=str(origin) if origin else None,
                tokens=list(permission_tokens),
            )
            raise ImportVerificationError(message)

        if spec.origin in {"built-in", "frozen"} or origin is None:
            return _ImportDecision(
                module=module,
                origin=None,
                status="builtin",
                manifest_entry=None,
                computed_hash=None,
            )

        if not origin.exists() or not origin.is_file():
            message = f"Module {module!r} has no importable file"
            self._record_blocked(
                module,
                session_id,
                sandbox,
                reason="missing_file",
                details=message,
                origin=str(origin),
            )
            raise ImportVerificationError(message)

        computed_hash = compute_file_hash(origin)

        if self._is_stdlib_path(origin):
            return _ImportDecision(
                module=module,
                origin=origin,
                status="external",
                manifest_entry=None,
                computed_hash=computed_hash,
            )

        if permission_manager and not permission_manager.is_allowed(module, permission_tokens):
            message = "Module access denied for capability tokens"
            self._record_blocked(
                module,
                session_id,
                sandbox,
                reason="permission_denied",
                details=message,
                origin=str(origin),
                computed_hash=computed_hash,
                tokens=list(permission_tokens),
            )
            raise ImportVerificationError(message)

        manifest_entry = self._manifest.get(module)
        if not manifest_entry:
            relative = self._relative_path(origin)
            if relative:
                manifest_entry = self._manifest.get(relative)

        if manifest_entry is None:
            message = f"Module {module!r} is not approved for import"
            self._record_blocked(
                module,
                session_id,
                sandbox,
                reason="missing_manifest",
                details=message,
                origin=str(origin),
                computed_hash=computed_hash,
                tokens=list(permission_tokens),
            )
            raise ImportVerificationError(message)

        expected_hash = manifest_entry.get("hash")
        if expected_hash != computed_hash:
            message = (
                f"Hash mismatch for module {module!r}: expected {expected_hash}, got {computed_hash}"
            )
            self._record_blocked(
                module,
                session_id,
                sandbox,
                reason="hash_mismatch",
                details=message,
                origin=str(origin),
                manifest_hash=expected_hash,
                computed_hash=computed_hash,
                tokens=list(permission_tokens),
            )
            raise ImportVerificationError(message)

        signing_key = manifest_entry.get("signing_key")
        signature_hex = manifest_entry.get("signature")

        if not signing_key or not signature_hex:
            if self._signing_keys:
                message = f"Module {module!r} is missing signing metadata"
                self._record_blocked(
                    module,
                    session_id,
                    sandbox,
                    reason="missing_signature",
                    details=message,
                    origin=str(origin),
                    manifest_hash=expected_hash,
                    computed_hash=computed_hash,
                    tokens=list(permission_tokens),
                    signing_key=signing_key,
                )
                raise ImportVerificationError(message)
            return _ImportDecision(
                module=module,
                origin=origin,
                status="verified",
                manifest_entry=manifest_entry,
                computed_hash=computed_hash,
                signing_key=None,
                signature_validated=False,
            )

        signature_error = self._verify_signature(origin, signing_key, signature_hex)
        if signature_error:
            message = (
                f"Signature verification failed for module {module!r}: {signature_error}"
            )
            self._record_blocked(
                module,
                session_id,
                sandbox,
                reason="signature_invalid",
                details=message,
                origin=str(origin),
                manifest_hash=expected_hash,
                computed_hash=computed_hash,
                tokens=list(permission_tokens),
                signing_key=signing_key,
            )
            raise ImportVerificationError(message)

        return _ImportDecision(
            module=module,
            origin=origin,
            status="verified",
            manifest_entry=manifest_entry,
            computed_hash=computed_hash,
            signing_key=signing_key,
            signature_validated=True,
        )

    def _verify_signature(
        self,
        origin: Path,
        key_id: str,
        signature_hex: str,
    ) -> Optional[str]:
        key_hex = self._signing_keys.get(key_id)
        if not key_hex:
            return f"unknown signing key {key_id!r}"
        try:
            key_bytes = bytes.fromhex(key_hex)
        except ValueError:
            return f"signing key {key_id!r} is not valid hex data"
        try:
            signature_bytes = bytes.fromhex(signature_hex)
        except ValueError:
            return "signature is not valid hex data"

        try:
            data = origin.read_bytes()
        except OSError as exc:  # pragma: no cover - defensive IO failure
            return f"failed to read module bytes: {exc}"

        try:
            public_key = Ed25519PublicKey.from_public_bytes(key_bytes)
        except ValueError:
            return f"signing key {key_id!r} is not a valid Ed25519 key"

        try:
            public_key.verify(signature_bytes, data)
        except InvalidSignature:
            return "signature does not match module bytes"
        return None

    # ------------------------------------------------------------------
    def _record_event(
        self,
        decision: _ImportDecision,
        session_id: str,
        sandbox: TokenSandbox,
        *,
        outcome: str,
        error: Optional[str] = None,
    ) -> None:
        payload: Dict[str, object] = {
            "session_id": session_id,
            "sandbox_id": sandbox.sandbox_id,
            "module": decision.module,
            "status": decision.status,
            "outcome": outcome,
        }
        if decision.origin is not None:
            payload["origin"] = str(decision.origin)
        if decision.computed_hash is not None:
            payload["computed_hash"] = decision.computed_hash
        if decision.manifest_entry is not None:
            payload["manifest_hash"] = decision.manifest_entry.get("hash")
            payload["manifest_path"] = decision.manifest_entry.get("path")
        if decision.signing_key is not None:
            payload["signing_key"] = decision.signing_key
        payload["signature_validated"] = decision.signature_validated
        if error is not None:
            payload["error"] = error
        if sandbox.metadata.get("capabilities"):
            payload["capabilities"] = list(sandbox.metadata.get("capabilities", []))

        kind = {
            "builtin": "python_vm_import_builtin",
            "external": "python_vm_import_external",
            "verified": "python_vm_import_verified",
        }.get(decision.status, "python_vm_import_event")

        event = self._ledger.record_event({"kind": kind, **payload})
        self._append_import_metadata(sandbox, {**payload, "event_id": event.get("event_id")})

    # ------------------------------------------------------------------
    def _record_blocked(
        self,
        module: str,
        session_id: str,
        sandbox: TokenSandbox,
        *,
        reason: str,
        details: str,
        origin: Optional[str] = None,
        manifest_hash: Optional[str] = None,
        computed_hash: Optional[str] = None,
        tokens: Optional[Sequence[str]] = None,
        signing_key: Optional[str] = None,
    ) -> None:
        payload: Dict[str, object] = {
            "kind": "python_vm_import_blocked",
            "session_id": session_id,
            "sandbox_id": sandbox.sandbox_id,
            "module": module,
            "reason": reason,
            "details": details,
        }
        if origin is not None:
            payload["origin"] = origin
        if manifest_hash is not None:
            payload["manifest_hash"] = manifest_hash
        if computed_hash is not None:
            payload["computed_hash"] = computed_hash
        if tokens is not None:
            payload["permission_tokens"] = list(tokens)
        if signing_key is not None:
            payload["signing_key"] = signing_key

        event = self._ledger.record_event(payload)
        self._append_import_metadata(
            sandbox,
            {
                "module": module,
                "status": "blocked",
                "reason": reason,
                "details": details,
                "event_id": event.get("event_id"),
                "permission_tokens": list(tokens) if tokens is not None else None,
            },
        )

    # ------------------------------------------------------------------
    def _append_import_metadata(self, sandbox: TokenSandbox, entry: Dict[str, object]) -> None:
        with self._lock:
            imports = list(sandbox.metadata.get("imports", []))
            imports.append(entry)
            sandbox._update_metadata({"imports": imports})

    def _relative_path(self, path: Path) -> Optional[str]:
        try:
            return str(path.resolve().relative_to(self._root))
        except ValueError:
            return None

    def _is_network_module(self, name: str) -> bool:
        for prefix in self._safe_deny_prefixes:
            if name == prefix or name.startswith(prefix + '.'):
                return True
        return False

    def _is_stdlib_path(self, path: Path) -> bool:
        resolved = path.resolve()
        for stdlib_path in self._stdlib_paths:
            try:
                resolved.relative_to(stdlib_path)
            except ValueError:
                continue
            else:
                return True
        return False


__all__ = [
    "DeterministicImportVerifier",
    "ImportVerificationError",
    "compute_file_hash",
]
