from __future__ import annotations

import json
from pathlib import Path

import pytest

from cli.command_shell import (
    CommandInvocation,
    ShellSession,
    kernel_performance_command,
)


def test_kernel_performance_records_and_summarizes(tmp_path: Path) -> None:
    shell = ShellSession(tmp_path)
    try:
        record_result = kernel_performance_command(
            shell,
            CommandInvocation(
                name="kernel-performance",
                args=[
                    "--json",
                    "record",
                    "12.5",
                    "20480",
                    "4096",
                    "--component",
                    "gpu",
                    "--notes",
                    "baseline run",
                ],
            ),
        )
        summary_result = kernel_performance_command(
            shell,
            CommandInvocation(
                name="kernel-performance",
                args=["--json", "summary"],
            ),
        )
        list_result = kernel_performance_command(
            shell,
            CommandInvocation(
                name="kernel-performance",
                args=["--json", "list"],
            ),
        )
    finally:
        shell.close()

    record_payload = json.loads(record_result.stdout)
    sample = record_payload["sample"]
    assert sample["sample_id"] == 1
    assert sample["component"] == "gpu"
    assert sample["notes"] == "baseline run"

    summary_payload = json.loads(summary_result.stdout)
    summary = summary_payload["summary"]
    assert summary["samples"] == 1
    assert summary["total_energy_joules"] == pytest.approx(12.5)
    assert summary["peak_memory_kb"] == 20480
    assert summary["total_io_bytes"] == 4096
    assert summary["components"]["gpu"]["samples"] == 1

    list_payload = json.loads(list_result.stdout)
    listed = list_payload["samples"]
    assert len(listed) == 1
    assert listed[0]["component"] == "gpu"

    ledger_path = tmp_path / "cli" / "data" / "snapshot_ledger.jsonl"
    lines = ledger_path.read_text(encoding="utf-8").splitlines()
    parsed = (json.loads(line) for line in reversed(lines))
    entry = next(item for item in parsed if item.get("kind") == "kernel_performance_recorded")
    assert entry["energy_joules"] == pytest.approx(12.5)
    assert entry["component"] == "gpu"


def test_kernel_performance_rejects_negative_energy(tmp_path: Path) -> None:
    shell = ShellSession(tmp_path)
    try:
        result = kernel_performance_command(
            shell,
            CommandInvocation(
                name="kernel-performance",
                args=[
                    "record",
                    "-1.0",
                    "1024",
                    "256",
                ],
            ),
        )
    finally:
        shell.close()

    assert result.status == 1
    assert "Energy must be non-negative" in result.stderr
