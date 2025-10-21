from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from cli.ledger_inspect import LedgerInspector
from cli.llm_adapter import DeterministicLLMAdapter
from cli.model_registry import ModelRecord
from cli.time_travel import TimeTravelDebugger


def _record(name: str = "demo") -> ModelRecord:
    return ModelRecord(
        name=name,
        source="local",
        provider="unittest",
        manifest=None,
        installed_at=datetime.now(timezone.utc).isoformat(),
        metadata={},
        capability="cap.demo",
    )


def test_llm_adapter_determinism() -> None:
    adapter = DeterministicLLMAdapter(salt="unit-test")
    record = _record()
    prompt = "deterministic prompt"

    result_one = adapter.generate(record, prompt)
    result_two = adapter.generate(record, prompt)

    assert result_one.digest == result_two.digest
    assert result_one.seed_material == result_two.seed_material


def test_time_travel_diff(tmp_path: Path) -> None:
    debugger = TimeTravelDebugger(tmp_path)
    state_dir = tmp_path / "cli" / "python_vm" / "snapshots" / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "00000001.json").write_text(json.dumps({"alpha": 1, "beta": 2}))
    (state_dir / "00000002.json").write_text(json.dumps({"alpha": 1, "gamma": 3}))

    snapshots = debugger.available_snapshots()
    assert snapshots == [1, 2]

    diff = debugger.diff(1, 2)
    assert diff["added"] == {"gamma": 3}
    assert diff["removed"] == {"beta": 2}
    assert diff["changed"] == {}


def test_ledger_inspector_summary(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.jsonl"
    entries = [
        {"event_id": "e1", "kind": "alpha", "ts": "2024-01-01T00:00:00Z"},
        {"event_id": "e2", "kind": "alpha", "ts": "2024-01-01T00:01:00Z"},
        {"event_id": "e3", "kind": "beta", "ts": "2024-01-01T00:02:00Z"},
    ]
    ledger.parent.mkdir(parents=True, exist_ok=True)
    ledger.write_text("\n".join(json.dumps(entry) for entry in entries) + "\n")

    inspector = LedgerInspector(ledger)
    summary = inspector.summary()
    assert summary["total_events"] == 3
    assert summary["by_kind"] == {"alpha": 2, "beta": 1}
    tail = inspector.tail(limit=2)
    assert len(tail) == 2
    assert tail[-1]["event_id"] == "e3"
