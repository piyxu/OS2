import json
from pathlib import Path

from cli.command_shell import (
    CommandInvocation,
    ShellSession,
    self_task_review_command,
)


def _read_ledger(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_self_task_review_records_skipped_and_active_events(tmp_path: Path) -> None:
    shell = ShellSession(tmp_path)
    try:
        inactive = self_task_review_command(
            shell,
            CommandInvocation(
                name="self-task-review",
                args=[
                    "--json",
                    "record",
                    "codex",
                    "task-1",
                    "success",
                    "42",
                    "hash-inactive",
                ],
            ),
        )
        enable = self_task_review_command(
            shell,
            CommandInvocation(
                name="self-task-review",
                args=["--json", "enable", "codex", "--api-key", "codex-key"],
            ),
        )
        active = self_task_review_command(
            shell,
            CommandInvocation(
                name="self-task-review",
                args=[
                    "--json",
                    "record",
                    "codex",
                    "task-2",
                    "completed",
                    "84",
                    "hash-active",
                ],
            ),
        )
        listing = self_task_review_command(
            shell,
            CommandInvocation(name="self-task-review", args=["--json", "list"]),
        )
    finally:
        shell.close()

    inactive_payload = json.loads(inactive.stdout)
    assert inactive_payload["ledger_event"]["status"] == "skipped"

    enable_payload = json.loads(enable.stdout)
    assert enable_payload["provider"]["active"] is True
    assert enable_payload["provider"]["codex_api_connected"] is True

    active_payload = json.loads(active.stdout)
    assert active_payload["ledger_event"]["status"] == "completed"
    assert active_payload["provider"]["last_response_hash"] == "hash-active"
    assert active_payload["provider"]["success_count"] == 1

    listing_payload = json.loads(listing.stdout)
    codex_entry = next(
        item for item in listing_payload["providers"] if item["provider_name"] == "codex"
    )
    assert codex_entry["active"] is True
    assert codex_entry["codex_api_connected"] is True
    assert codex_entry["status_counts"]["skipped"] == 1

    ledger_path = tmp_path / "cli" / "data" / "snapshot_ledger.jsonl"
    entries = _read_ledger(ledger_path)
    kinds = [entry.get("kind") for entry in entries]
    assert kinds == [
        "self_task_review_task_event",
        "self_task_review_provider_status",
        "self_task_review_task_event",
    ]
    assert entries[0]["status"] == "skipped"
    assert entries[2]["status"] == "completed"


def test_disable_provider_logs_skipped_event(tmp_path: Path) -> None:
    shell = ShellSession(tmp_path)
    try:
        self_task_review_command(
            shell,
            CommandInvocation(
                name="self-task-review",
                args=["enable", "codex", "--api-key", "codex-key"],
            ),
        )
        self_task_review_command(
            shell,
            CommandInvocation(name="self-task-review", args=["disable", "codex"]),
        )
        skipped = self_task_review_command(
            shell,
            CommandInvocation(
                name="self-task-review",
                args=["--json", "record", "codex", "task-3", "failed", "12", "hash-skip"],
            ),
        )
    finally:
        shell.close()

    skipped_payload = json.loads(skipped.stdout)
    assert skipped_payload["ledger_event"]["status"] == "skipped"
    assert skipped_payload["provider"]["active"] is False
    assert skipped_payload["provider"]["status_counts"]["skipped"] >= 1

    ledger_path = tmp_path / "cli" / "data" / "snapshot_ledger.jsonl"
    entries = _read_ledger(ledger_path)
    assert entries[-1]["status"] == "skipped"
    assert entries[-1]["provider_name"] == "codex"
