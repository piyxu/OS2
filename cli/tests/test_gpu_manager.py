import json
from pathlib import Path

import pytest

from cli.gpu_manager import GPUAccessError, GPUAccessManager
from cli.snapshot_ledger import SnapshotLedger


def test_gpu_manager_acquire_release(tmp_path: Path) -> None:
    root = tmp_path
    ledger_path = root / "cli" / "data" / "snapshot_ledger.jsonl"
    snapshot_ledger = SnapshotLedger(ledger_path)
    manager = GPUAccessManager(root, snapshot_ledger)

    lease = manager.acquire(capability="cap.model.demo", backend="cuda", device="cuda:0", model="demo")
    assert lease.lease_id
    assert lease.granted_event["kind"] == "gpu_access_granted"

    release_event = manager.release(lease.lease_id, tokens=42)
    assert release_event["kind"] == "gpu_access_released"
    assert release_event["tokens"] == 42

    log_path = root / "cli" / "data" / "gpu_access.jsonl"
    lines = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line]
    kinds = [entry.get("kind") for entry in lines]
    assert kinds == ["gpu_access_granted", "gpu_access_released"]


def test_gpu_manager_unknown_lease(tmp_path: Path) -> None:
    root = tmp_path
    snapshot_ledger = SnapshotLedger(root / "cli" / "data" / "snapshot_ledger.jsonl")
    manager = GPUAccessManager(root, snapshot_ledger)

    with pytest.raises(GPUAccessError):
        manager.release("missing-lease")
