import pytest

from cli.snapshot_ledger import SnapshotLedger, SnapshotLedgerError


def test_snapshot_ledger_read_only(tmp_path) -> None:
    ledger = SnapshotLedger(tmp_path / "ledger.jsonl")
    event = ledger.record_event({"kind": "test_event"})
    assert event["kind"] == "test_event"

    ledger.set_read_only(True)
    with pytest.raises(SnapshotLedgerError):
        ledger.record_event({"kind": "should_fail"})

    ledger.set_read_only(False)
    second = ledger.record_event({"kind": "second"})
    assert second["kind"] == "second"
