from __future__ import annotations

import json
from pathlib import Path

from cli.documentation import DocumentationPublisher
from cli.snapshot_ledger import SnapshotLedger


def test_publish_shell_manual(tmp_path: Path) -> None:
    ledger = SnapshotLedger(tmp_path / "ledger.jsonl")
    docs_root = tmp_path / "docs"
    docs_root.mkdir()
    publisher = DocumentationPublisher(tmp_path, ledger)

    result = publisher.publish_shell_manual()

    manual_path = result["path"]
    assert manual_path.exists()
    assert "PIYXU Deterministic Shell Technical Guide" in manual_path.read_text(encoding="utf-8")
    event = result["event"]
    assert event["kind"] == "documentation_published"


def test_release_workflow(tmp_path: Path) -> None:
    ledger = SnapshotLedger(tmp_path / "ledger.jsonl")
    publisher = DocumentationPublisher(tmp_path, ledger)

    result = publisher.publish_release_workflow()

    workflow_path = result["path"]
    assert workflow_path.exists()
    body = workflow_path.read_text(encoding="utf-8")
    assert "Deterministic Release Workflow" in body
    assert "document-module-tree --json" in body
    assert result["event"]["kind"] == "release_workflow_documented"


def test_module_tree(tmp_path: Path) -> None:
    modules_dir = tmp_path / "cli" / "modules"
    modules_dir.mkdir(parents=True)
    manifest = {
        "name": "alpha",
        "token_id": "t-alpha",
        "commands": [{"name": "noop"}],
    }
    (modules_dir / "alpha.json").write_text(json.dumps(manifest), encoding="utf-8")

    ledger = SnapshotLedger(tmp_path / "ledger.jsonl")
    publisher = DocumentationPublisher(tmp_path, ledger)
    result = publisher.document_module_tree()

    module_doc = result["path"]
    assert module_doc.exists()
    lines = module_doc.read_text(encoding="utf-8")
    assert "| alpha | t-alpha | 1 |" in lines
    assert result["event"]["module_count"] == 1


def test_ready_flag(tmp_path: Path) -> None:
    ledger = SnapshotLedger(tmp_path / "ledger.jsonl")
    publisher = DocumentationPublisher(tmp_path, ledger)

    result = publisher.set_ready_flag(ready=True)

    state_path = result["path"]
    assert state_path.exists()
    data = json.loads(state_path.read_text(encoding="utf-8"))
    assert data["ready_for_next_evolution"] is True
    assert result["event"]["ready"] is True
