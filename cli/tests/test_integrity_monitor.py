from pathlib import Path

from cli.integrity_monitor import IntegrityMonitor
from cli.snapshot_ledger import SnapshotLedger


def test_integrity_monitor_hashes_files(tmp_path: Path) -> None:
    ledger = SnapshotLedger(tmp_path / "ledger.jsonl")
    monitor = IntegrityMonitor(tmp_path, ledger)

    file_path = tmp_path / "artifact.txt"
    file_path.write_text("payload", encoding="utf-8")

    report = monitor.run_check(label="test", paths=[file_path])
    assert report.files_hashed == 1
    assert report.digest
