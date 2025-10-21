import hashlib
import json
from pathlib import Path

import cli.command_shell as command_shell
from cli.command_shell import ShellSession


def _register_all_commands(shell: ShellSession) -> None:
    for obj in command_shell.__dict__.values():
        if callable(obj) and hasattr(obj, "__command_definition__"):
            shell.register(obj.__command_definition__)


def _write_unsigned_module(path: Path) -> None:
    spec = {
        "name": "secure-tools",
        "commands": [
            {
                "name": "secure-echo",
                "summary": "Echo arguments deterministically",
                "usage": "secure-echo <text>",
                "capabilities": ["basic"],
                "response": "{args}",
            }
        ],
    }
    path.write_text(json.dumps(spec, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_signed_module(path: Path, shell: ShellSession) -> None:
    spec = {
        "name": "secure-tools",
        "token_id": "default",
        "signature_algorithm": "hmac-sha256",
        "commands": [
            {
                "name": "secure-echo",
                "summary": "Echo arguments deterministically",
                "usage": "secure-echo <text>",
                "capabilities": ["basic"],
                "response": "{args}",
            }
        ],
    }
    canonical = dict(spec)
    canonical.pop("signature", None)
    canonical.pop("token_id", None)
    canonical.pop("signature_algorithm", None)
    digest = hashlib.sha256(
        json.dumps(canonical, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    signature = shell.signature_verifier.sign_digest(digest, key_id="default")
    spec["signature"] = signature
    path.write_text(json.dumps(spec, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")



def test_load_modules_requires_signature(tmp_path: Path) -> None:
    shell = ShellSession(tmp_path)
    _register_all_commands(shell)
    modules_dir = tmp_path / "cli" / "modules"
    modules_dir.mkdir(parents=True, exist_ok=True)
    module_path = modules_dir / "secure-tools.json"
    try:
        _write_unsigned_module(module_path)
        result = shell.run_line("load-modules")
        assert result.status == 1
        assert "missing signature" in result.stderr

        _write_signed_module(module_path, shell)
        result = shell.run_line("load-modules")
        assert result.status == 0
        assert "Loaded modules" in result.stdout

        echo = shell.run_line("secure-echo hello")
        assert echo.status == 0
        assert echo.stdout.strip() == "hello"
    finally:
        shell.close()



def test_admin_commands_require_snapshot_auth(tmp_path: Path) -> None:
    shell = ShellSession(tmp_path)
    _register_all_commands(shell)
    try:
        result = shell.run_line("module-perms list")
        assert result.status == 1
        assert "require snapshot authentication" in result.stderr

        auth = shell.run_line("snapshot-auth 42 --reason test --json")
        assert auth.status == 0
        payload = json.loads(auth.stdout)
        assert payload["snapshot_id"] == 42

        result = shell.run_line("module-perms --json list")
        assert result.status == 0
        permissions = json.loads(result.stdout)
        assert permissions["permissions"]["default"] == ["__all__"]
    finally:
        shell.close()



def test_hash_ledger_command_lock_unlock(tmp_path: Path) -> None:
    shell = ShellSession(tmp_path)
    _register_all_commands(shell)
    try:
        shell.run_line("snapshot-auth 7")

        status = shell.run_line("hash-ledger status")
        assert status.status == 0
        assert "writable" in status.stdout

        lock = shell.run_line("hash-ledger lock")
        assert lock.status == 0
        assert shell.snapshot_ledger.is_read_only()
        assert "read-only" in lock.stdout

        unlock = shell.run_line("hash-ledger unlock")
        assert unlock.status == 0
        assert not shell.snapshot_ledger.is_read_only()

        ledger_path = tmp_path / "cli" / "data" / "snapshot_ledger.jsonl"
        entries = [json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines()]
        modes = [entry.get("mode") for entry in entries if entry.get("kind") == "snapshot_ledger_mode_changed"]
        assert modes[-2:] == ["read_only", "writable"]
    finally:
        shell.close()
