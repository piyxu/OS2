from __future__ import annotations

import json
from pathlib import Path

from cli.kernel_metrics import KernelTaskAnalyzer


def _write_events(path: Path, events: list[dict]) -> None:
    payload = "\n".join(json.dumps(event) for event in events if event)
    path.write_text(payload + "\n", encoding="utf-8")


def test_kernel_task_analyzer_computes_rates(tmp_path: Path) -> None:
    log_path = tmp_path / "kernel_events.jsonl"
    _write_events(
        log_path,
        [
            {
                "timestamp": 1,
                "sequence": 1,
                "label": "model_inference",
                "detail": {"status": "completed"},
            },
            {
                "timestamp": 2,
                "sequence": 2,
                "label": "model_inference",
                "detail": {"status": "failed"},
            },
            {
                "timestamp": 3,
                "sequence": 3,
                "label": "python_vm_session",
                "detail": {"status": "ok"},
            },
            {
                "timestamp": 4,
                "sequence": 4,
                "label": "python_vm_session",
                "detail": {"status": "skipped"},
            },
            {
                "timestamp": 5,
                "sequence": 5,
                "label": "python_vm_async_task",
                "detail": {"status": "timeout"},
            },
        ],
    )

    analyzer = KernelTaskAnalyzer(log_path)
    summary = analyzer.compute()
    data = summary.to_dict()

    assert summary.total == 5
    assert summary.success == 2
    assert summary.failure == 2
    assert summary.skipped == 1
    assert summary.attempted == 4
    assert summary.success_rate == 0.5
    assert data["by_label"]["model_inference"]["success"] == 1
    assert data["by_label"]["python_vm_session"]["skipped"] == 1
    assert summary.last_timestamp == 5
    assert summary.last_sequence == 5


def test_kernel_task_analyzer_empty(tmp_path: Path) -> None:
    analyzer = KernelTaskAnalyzer(tmp_path / "missing.jsonl")
    summary = analyzer.compute()

    assert summary.total == 0
    assert summary.attempted == 0
    assert summary.success_rate == 0.0
    assert summary.by_label == {}
