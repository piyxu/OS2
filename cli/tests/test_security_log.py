from pathlib import Path

from cli.ai_replay import AIReplayManager
from cli.model_registry import ModelRegistry
from cli.security_log import SecurityLogManager
from cli.snapshot_ledger import SnapshotLedger


def test_security_log_integration(tmp_path: Path) -> None:
    ledger = SnapshotLedger(tmp_path / "cli" / "data" / "snapshot_ledger.jsonl")
    registry = ModelRegistry(tmp_path / "cli" / "data" / "models.json")
    manager = SecurityLogManager(tmp_path, ledger)

    event = manager.record(event_type="audit", message="ok", severity="info")
    assert event.event_type == "audit"

    events = list(manager.iter_events())
    assert len(events) == 1

    replay = AIReplayManager(tmp_path, ledger, registry, llm_adapter=None)  # type: ignore[arg-type]
    result = manager.integrate_with_replay(replay)
    assert result["count"] == 1
