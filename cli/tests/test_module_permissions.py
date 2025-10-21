from pathlib import Path

from cli.module_permissions import ModulePermissionRegistry, ModulePermissionError
from cli.snapshot_ledger import SnapshotLedger


def test_module_permissions_grant_and_revoke(tmp_path: Path) -> None:
    ledger = SnapshotLedger(tmp_path / "ledger.jsonl")
    registry = ModulePermissionRegistry(tmp_path, ledger)

    registry.revoke("default", "__all__")

    result = registry.grant("token-alpha", "package.module")
    assert registry.is_allowed("package.module", ["token-alpha"])
    event = result["ledger_event"]
    assert event is not None
    assert event["kind"] == "python_module_permission_granted"

    duplicate = registry.grant("token-alpha", "package.module")
    assert duplicate["ledger_event"] is None

    revoke = registry.revoke("token-alpha", "package.module")
    assert not registry.is_allowed("package.module", ["token-alpha"])
    revoke_event = revoke["ledger_event"]
    assert revoke_event is not None
    assert revoke_event["kind"] == "python_module_permission_revoked"


def test_module_permissions_errors(tmp_path: Path) -> None:
    ledger = SnapshotLedger(tmp_path / "ledger.jsonl")
    registry = ModulePermissionRegistry(tmp_path, ledger)

    try:
        registry.revoke("missing", "example")
    except ModulePermissionError:
        pass
    else:  # pragma: no cover - defensive
        raise AssertionError("Expected ModulePermissionError when revoking unknown module")
