from pathlib import Path

from cli.backup_manager import SecureBackupManager
from cli.signature import SignatureVerifier
from cli.snapshot_ledger import SnapshotLedger


def test_secure_backup_creates_signed_backup(tmp_path: Path) -> None:
    ledger = SnapshotLedger(tmp_path / "cli" / "data" / "snapshot_ledger.jsonl")
    verifier = SignatureVerifier({"default": "secret"}, default_key="default")
    manager = SecureBackupManager(tmp_path, ledger, verifier)

    target = tmp_path / "important.txt"
    target.write_text("data", encoding="utf-8")

    summary = manager.create_backup(label="daily", include=[target])
    assert summary.backup_dir.exists()
    assert summary.signature
    assert any(path.name == "important.txt" for path in summary.files)
