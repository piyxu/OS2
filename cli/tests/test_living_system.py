from __future__ import annotations

import json
from pathlib import Path

import pytest

from cli.command_shell import (
    CommandInvocation,
    ShellSession,
    living_system_command,
)
from cli.kernel_performance import KernelPerformanceMonitor
from cli.living_system import (
    LivingDeterministicSystemError,
    LivingDeterministicSystemManager,
)
from cli.self_feedback import SelfFeedbackAnalyzer
from cli.snapshot_benchmark import SnapshotBenchmarkManager
from cli.snapshot_ledger import SnapshotLedger


class _StubAnalyzer:
    class _Result:
        def to_dict(self) -> dict:
            return {
                "total": 1,
                "attempted": 1,
                "success": 1,
                "failure": 0,
                "skipped": 0,
                "success_rate": 1.0,
                "by_label": {},
                "last_timestamp": 1,
                "last_sequence": 1,
            }

    def compute(self) -> "_StubAnalyzer._Result":  # pragma: no cover - simple stub
        return self._Result()


def _prepare_kernel_events(root: Path) -> None:
    events_dir = root / "rust" / "os2-kernel" / "logs"
    events_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "label": "task",
        "detail": {"status": "success"},
        "timestamp": 1,
        "sequence": 1,
    }
    (events_dir / "kernel_events.jsonl").write_text(
        json.dumps(payload, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _prepare_module_definition(root: Path) -> None:
    modules_dir = root / "cli" / "modules"
    modules_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "name": "core",
        "commands": [
            {"name": "ping", "summary": "Ping the kernel", "response": "pong"},
        ],
    }
    (modules_dir / "core.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def test_living_system_transition_requires_ready_components(tmp_path: Path) -> None:
    ledger = SnapshotLedger(tmp_path / "cli" / "data" / "snapshot_ledger.jsonl")
    manager = LivingDeterministicSystemManager(tmp_path, ledger)

    with pytest.raises(LivingDeterministicSystemError):
        manager.transition(operator="tester")


def test_living_system_transition_records_state(tmp_path: Path) -> None:
    ledger_path = tmp_path / "cli" / "data" / "snapshot_ledger.jsonl"
    ledger = SnapshotLedger(ledger_path)
    _prepare_module_definition(tmp_path)

    performance = KernelPerformanceMonitor(tmp_path, ledger)
    performance.record(energy_joules=1.0, memory_kb=256, io_bytes=128, component="kernel")

    benchmark = SnapshotBenchmarkManager(tmp_path, ledger, analyzer_factory=_StubAnalyzer)
    benchmark.evaluate(force=True)

    feedback = SelfFeedbackAnalyzer(tmp_path, ledger)
    feedback.ingest(
        {
            "command": "module-prune",
            "status": 0,
            "stdout": "ok",
            "stderr": "",
            "args": [],
        }
    )

    manager = LivingDeterministicSystemManager(tmp_path, ledger)
    state = manager.transition(operator="release", notes="Stage 4.10")

    assert state["current_stage"] == "living"
    readiness = state["readiness"]
    assert readiness["ready_components"] >= readiness["minimum_ready"]
    assert readiness["state"] in {"ready", "stabilizing"}
    assert state["components"]["kernel_performance"]["state"] == "ready"
    assert state["components"]["snapshot_benchmarks"]["state"] == "ready"
    assert state["components"]["self_feedback"]["state"] == "ready"
    assert state["ledger_event"]["kind"] == "living_system_transition"

    status_payload = manager.status(refresh=True)
    assert "observation" in status_payload
    assert status_payload["state"]["current_stage"] == "living"

    saved_state = json.loads(
        (tmp_path / "cli" / "data" / "living_system_state.json").read_text(encoding="utf-8")
    )
    assert saved_state["current_stage"] == "living"


def test_living_system_command_transition(tmp_path: Path) -> None:
    _prepare_module_definition(tmp_path)
    shell = ShellSession(tmp_path)
    try:
        shell.kernel_performance.record(energy_joules=2.0, memory_kb=512, io_bytes=256, component="kernel")
        _prepare_kernel_events(tmp_path)
        shell.snapshot_benchmarks.evaluate(force=True)
        shell.self_feedback.ingest(
            {
                "command": "kernel-performance",
                "status": 0,
                "stdout": "ok",
                "stderr": "",
                "args": [],
            }
        )

        result = living_system_command(
            shell,
            CommandInvocation(
                name="living-system",
                args=[
                    "transition",
                    "--json",
                    "--operator",
                    "orchestrator",
                    "--notes",
                    "Activating stage 4.10",
                ],
            ),
        )
        payload = json.loads(result.stdout)
        assert payload["current_stage"] == "living"
        assert payload["readiness"]["ready_components"] >= payload["readiness"]["minimum_ready"]

        status = living_system_command(
            shell,
            CommandInvocation(name="living-system", args=["status", "--refresh", "--json"]),
        )
        status_payload = json.loads(status.stdout)
        assert "observation" in status_payload
        assert status_payload["state"]["current_stage"] == "living"
    finally:
        shell.close()
