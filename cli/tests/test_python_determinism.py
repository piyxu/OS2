import json
from pathlib import Path

from cli.command_shell import (
    CommandInvocation,
    ShellSession,
    python_command,
    python_verify_command,
)


def _read_ledger(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_python_verify_command(tmp_path: Path) -> None:
    shell = ShellSession(tmp_path)
    try:
        result = python_command(
            shell, CommandInvocation(name="python", args=["-c", "print('determinism')"])
        )
        assert result.status == 0

        verify = python_verify_command(
            shell, CommandInvocation(name="python-verify", args=["--json"])
        )
    finally:
        shell.close()

    assert verify.status == 0
    payload = json.loads(verify.stdout)
    assert payload["checked_sessions"] >= 1
    assert payload["checked_sessions"] == payload["verified_sessions"]
    assert payload["failures"] == []

    ledger_path = tmp_path / "cli" / "data" / "snapshot_ledger.jsonl"
    kinds = [entry["kind"] for entry in _read_ledger(ledger_path)]
    assert "python_vm_replay_verified" in kinds
