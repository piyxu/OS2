import json
from pathlib import Path

from cli.command_shell import (
    CommandInvocation,
    ShellSession,
    snapshot_benchmarks_command,
)


def _write_kernel_events(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    events = [
        {
            "timestamp": 1,
            "sequence": 1,
            "label": "task",
            "detail": {"status": "success"},
        },
        {
            "timestamp": 2,
            "sequence": 2,
            "label": "task",
            "detail": {"status": "failure"},
        },
        {
            "timestamp": 3,
            "sequence": 3,
            "label": "task",
            "detail": {"status": "skipped"},
        },
    ]
    lines = "\n".join(json.dumps(event) for event in events) + "\n"
    path.write_text(lines, encoding="utf-8")


def _load_json(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def test_snapshot_benchmark_run_records_summary(tmp_path: Path) -> None:
    log_path = tmp_path / "rust" / "os2-kernel" / "logs" / "kernel_events.jsonl"
    _write_kernel_events(log_path)
    shell = ShellSession(tmp_path)
    try:
        result = snapshot_benchmarks_command(
            shell,
            CommandInvocation(name="snapshot-benchmarks", args=["--json", "run"]),
        )
    finally:
        shell.close()

    assert result.status == 0
    payload = json.loads(result.stdout)
    benchmark = payload["benchmark"]
    summary = benchmark["task_summary"]
    assert benchmark["benchmark_id"] == 1
    assert summary["total"] == 3
    assert summary["attempted"] == 2
    assert summary["success"] == 1

    data_path = tmp_path / "cli" / "data" / "snapshot_benchmarks.json"
    stored = _load_json(data_path)
    assert stored[0]["benchmark_id"] == 1
    assert stored[0]["task_summary"]["total"] == 3
    assert stored[0]["ledger_event_id"]

    ledger_path = tmp_path / "cli" / "data" / "snapshot_ledger.jsonl"
    lines = ledger_path.read_text(encoding="utf-8").splitlines()
    parsed = (json.loads(line) for line in reversed(lines))
    entry = next(item for item in parsed if item.get("kind") == "snapshot_benchmark_completed")
    assert entry["benchmark_id"] == 1


def test_snapshot_benchmark_enforces_interval(tmp_path: Path) -> None:
    log_path = tmp_path / "rust" / "os2-kernel" / "logs" / "kernel_events.jsonl"
    _write_kernel_events(log_path)
    shell = ShellSession(tmp_path)
    try:
        first = snapshot_benchmarks_command(
            shell,
            CommandInvocation(name="snapshot-benchmarks", args=["run"]),
        )
        second = snapshot_benchmarks_command(
            shell,
            CommandInvocation(name="snapshot-benchmarks", args=["run"]),
        )
    finally:
        shell.close()

    assert first.status == 0
    assert second.status == 1
    assert "last 24 hours" in second.stderr


def test_snapshot_benchmark_status_reports_window(tmp_path: Path) -> None:
    log_path = tmp_path / "rust" / "os2-kernel" / "logs" / "kernel_events.jsonl"
    _write_kernel_events(log_path)
    shell = ShellSession(tmp_path)
    try:
        initial = snapshot_benchmarks_command(
            shell,
            CommandInvocation(name="snapshot-benchmarks", args=["--json", "status"]),
        )
        snapshot_benchmarks_command(
            shell,
            CommandInvocation(name="snapshot-benchmarks", args=["run"]),
        )
        status_after = snapshot_benchmarks_command(
            shell,
            CommandInvocation(name="snapshot-benchmarks", args=["--json", "status"]),
        )
    finally:
        shell.close()

    initial_payload = json.loads(initial.stdout)
    assert initial_payload["last_benchmark"] is None

    status_payload = json.loads(status_after.stdout)
    assert status_payload["last_benchmark"]["benchmark_id"] == 1
    assert status_payload["next_allowed_at"]
    assert status_payload["seconds_until_next"] > 0
