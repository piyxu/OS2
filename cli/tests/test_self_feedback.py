from __future__ import annotations

import json
from pathlib import Path

from cli.command_shell import (
    CommandInvocation,
    ShellSession,
    self_feedback_command,
)
from cli.self_feedback import SelfFeedbackAnalyzer
from cli.snapshot_ledger import SnapshotLedger


def test_self_feedback_ingest_updates_summary(tmp_path: Path) -> None:
    ledger_path = tmp_path / "cli" / "data" / "snapshot_ledger.jsonl"
    ledger = SnapshotLedger(ledger_path)
    analyzer = SelfFeedbackAnalyzer(tmp_path, ledger, max_history=3)

    positive_payload = analyzer.ingest(
        {
            "command": "echo",
            "args": ["hello"],
            "status": 0,
            "stdout": "hello\n",
            "stderr": "",
            "timestamp": "2024-01-01T00:00:00Z",
        }
    )
    negative_payload = analyzer.ingest(
        {
            "command": "failing",
            "args": ["--flag"],
            "status": 2,
            "stdout": "",
            "stderr": "error: boom\n",
            "timestamp": "2024-01-01T00:01:00Z",
        }
    )

    summary = analyzer.summary()
    assert summary["total_interactions"] == 2
    assert summary["success_count"] == 1
    assert summary["failure_count"] == 1
    assert summary["positive_count"] == 1
    assert summary["negative_count"] == 1
    assert summary["commands"]["echo"]["success"] == 1
    assert summary["commands"]["failing"]["failure"] == 1

    recent = analyzer.recent_interactions()
    assert recent[0]["command"] == "failing"
    assert recent[1]["command"] == "echo"

    data_path = tmp_path / "cli" / "data" / "self_feedback.json"
    stored = json.loads(data_path.read_text(encoding="utf-8"))
    assert stored["total_interactions"] == 2
    assert stored["commands"]["echo"]["count"] == 1

    ledger_lines = ledger_path.read_text(encoding="utf-8").splitlines()
    parsed = (json.loads(line) for line in ledger_lines)
    kinds = [entry["kind"] for entry in parsed]
    assert "self_feedback_interaction_recorded" in kinds
    assert positive_payload["interaction"]["sentiment"] == "positive"
    assert negative_payload["interaction"]["sentiment"] == "negative"


def test_self_feedback_command_reports_summary_and_recent(tmp_path: Path) -> None:
    shell = ShellSession(tmp_path)
    try:
        shell.self_feedback.ingest(
            {
                "command": "list",
                "status": 0,
                "stdout": "ok\n",
                "stderr": "",
                "timestamp": "2024-01-01T00:00:00Z",
            }
        )
        shell.self_feedback.ingest(
            {
                "command": "list",
                "status": 1,
                "stdout": "",
                "stderr": "error\n",
                "timestamp": "2024-01-01T00:01:00Z",
            }
        )
        summary_result = self_feedback_command(
            shell,
            CommandInvocation(name="self-feedback", args=["--json", "summary"]),
        )
        recent_result = self_feedback_command(
            shell,
            CommandInvocation(
                name="self-feedback",
                args=["--json", "recent", "--limit", "1"],
            ),
        )
    finally:
        shell.close()

    assert summary_result.status == 0
    summary_payload = json.loads(summary_result.stdout)
    summary = summary_payload["summary"]
    assert summary["total_interactions"] >= 2
    assert summary["failure_count"] >= 1

    recent_payload = json.loads(recent_result.stdout)
    interactions = recent_payload["interactions"]
    assert len(interactions) == 1
    assert interactions[0]["command"] == "list"
