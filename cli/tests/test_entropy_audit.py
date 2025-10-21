from pathlib import Path

from cli.entropy_audit import EntropyAuditor
from cli.snapshot_ledger import SnapshotLedger


def test_entropy_audit_detects_deviation(tmp_path: Path) -> None:
    ledger = SnapshotLedger(tmp_path / "cli" / "data" / "snapshot_ledger.jsonl")
    auditor = EntropyAuditor(tmp_path, ledger)

    ledger.record_event({"kind": "download_entropy_captured", "entropy_bits": 16})
    ledger.record_event({"kind": "download_entropy_captured", "entropy_bits": 5})

    result = auditor.audit()
    assert result["total_events"] == 2
    assert len(result["deviations"]) == 1
