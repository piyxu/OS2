import json
from pathlib import Path

from cli.command_shell import (
    CommandInvocation,
    ShellSession,
    deterministic_recompile_command,
)


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_ledger(path: Path) -> list[dict]:
    if not path.exists():
        return []
    entries: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entries.append(json.loads(line))
    return entries


def test_deterministic_recompile_queue_and_approve(tmp_path: Path) -> None:
    tracked = tmp_path / "src" / "module.py"
    tracked.parent.mkdir(parents=True, exist_ok=True)
    tracked.write_text("print('deterministic')\n", encoding="utf-8")

    shell = ShellSession(tmp_path)
    try:
        queue_result = deterministic_recompile_command(
            shell,
            CommandInvocation(
                name="deterministic-recompile",
                args=[
                    "--json",
                    "queue",
                    "change-001",
                    "--path",
                    "src/module.py",
                    "--description",
                    "Recompile deterministic module",
                    "--tag",
                    "phase-4",
                ],
            ),
        )
        approve_result = deterministic_recompile_command(
            shell,
            CommandInvocation(
                name="deterministic-recompile",
                args=["--json", "approve", "change-001"],
            ),
        )
        pending_result = deterministic_recompile_command(
            shell,
            CommandInvocation(
                name="deterministic-recompile",
                args=["--json", "pending"],
            ),
        )
    finally:
        shell.close()

    queued_payload = json.loads(queue_result.stdout)
    assert queued_payload["queued_change"]["change_id"] == "change-001"
    assert queued_payload["queued_change"]["paths"] == ["src/module.py"]
    assert queued_payload["queued_change"]["tags"] == ["phase-4"]

    approved_payload = json.loads(approve_result.stdout)
    assert approved_payload["approved_change"]["change_id"] == "change-001"
    assert approved_payload["approved_change"]["reviewer"] == shell.user

    pending_payload = json.loads(pending_result.stdout)
    assert pending_payload["pending"] == []

    state_path = tmp_path / "cli" / "data" / "deterministic_recompile.json"
    state = _read_json(state_path)
    history = state.get("history", [])
    assert history and history[0]["change_id"] == "change-001"

    ledger_entries = _read_ledger(tmp_path / "cli" / "data" / "snapshot_ledger.jsonl")
    kinds = {entry.get("kind") for entry in ledger_entries}
    assert "deterministic_recompile_submitted" in kinds
    assert "deterministic_recompile_approved" in kinds


def test_deterministic_recompile_detects_hash_mismatch(tmp_path: Path) -> None:
    tracked = tmp_path / "src" / "module.py"
    tracked.parent.mkdir(parents=True, exist_ok=True)
    tracked.write_text("print('v1')\n", encoding="utf-8")

    shell = ShellSession(tmp_path)
    try:
        deterministic_recompile_command(
            shell,
            CommandInvocation(
                name="deterministic-recompile",
                args=[
                    "queue",
                    "change-xyz",
                    "--path",
                    "src/module.py",
                ],
            ),
        )
        tracked.write_text("print('v2')\n", encoding="utf-8")
        approve_result = deterministic_recompile_command(
            shell,
            CommandInvocation(
                name="deterministic-recompile",
                args=["approve", "change-xyz"],
            ),
        )
    finally:
        shell.close()

    assert approve_result.status == 1
    assert "mismatch" in approve_result.stderr.lower()

    state_path = tmp_path / "cli" / "data" / "deterministic_recompile.json"
    state = _read_json(state_path)
    pending = state.get("pending", [])
    assert pending and pending[0]["change_id"] == "change-xyz"
    assert pending[0]["last_outcome"]["status"] == "mismatch"

    ledger_entries = _read_ledger(tmp_path / "cli" / "data" / "snapshot_ledger.jsonl")
    mismatch_events = [
        entry for entry in ledger_entries if entry.get("kind") == "deterministic_recompile_mismatch"
    ]
    assert mismatch_events, "expected mismatch event recorded in ledger"
