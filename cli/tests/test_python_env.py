from __future__ import annotations

import json
from pathlib import Path

import pytest

from cli.command_shell import CommandInvocation, ShellSession, create_env_command


def _read_ledger(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_create_env_records_ledger_and_metadata(tmp_path: Path) -> None:
    shell = ShellSession(tmp_path)
    try:
        invocation = CommandInvocation(
            name="create-env",
            args=["dev-env", "--json", "--no-pip"],
        )
        result = create_env_command(shell, invocation)
    finally:
        shell.close()

    assert result.status == 0
    payload = json.loads(result.stdout)
    assert payload["environment_id"] == "dev-env"
    assert payload["status"] == "created"
    assert payload["ledger_event_ids"]["started"]
    assert payload["ledger_event_ids"]["created"]
    env_dir = tmp_path / "cli" / "python_vm" / "environments" / "dev-env"
    assert env_dir.exists()
    metadata = json.loads((env_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["status"] == "created"
    assert metadata["ledger_event_ids"]["created"] == payload["ledger_event_ids"]["created"]
    ledger_path = tmp_path / "cli" / "data" / "snapshot_ledger.jsonl"
    entries = _read_ledger(ledger_path)
    kinds = [entry["kind"] for entry in entries]
    assert kinds == ["python_vm_env_creation_started", "python_vm_env_created"]


def test_duplicate_environment_emits_failure(tmp_path: Path) -> None:
    shell = ShellSession(tmp_path)
    try:
        first = create_env_command(
            shell,
            CommandInvocation(name="create-env", args=["dev-env", "--no-pip"]),
        )
        second = create_env_command(
            shell,
            CommandInvocation(name="create-env", args=["dev-env", "--no-pip"]),
        )
    finally:
        shell.close()

    assert first.status == 0
    assert second.status == 1
    assert "already exists" in second.stderr
    ledger_path = tmp_path / "cli" / "data" / "snapshot_ledger.jsonl"
    entries = _read_ledger(ledger_path)
    kinds = [entry["kind"] for entry in entries]
    assert kinds[-1] == "python_vm_env_creation_failed"
    assert entries[-1]["status"] == "exists"


def test_environment_name_validation(tmp_path: Path) -> None:
    shell = ShellSession(tmp_path)
    try:
        result = create_env_command(
            shell,
            CommandInvocation(name="create-env", args=["../bad-name"]),
        )
    finally:
        shell.close()

    assert result.status == 1
    assert "Environment name" in result.stderr
    ledger_path = tmp_path / "cli" / "data" / "snapshot_ledger.jsonl"
    entries = _read_ledger(ledger_path)
    assert entries == []
