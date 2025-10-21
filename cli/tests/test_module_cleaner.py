import json
from pathlib import Path

from cli.command_shell import (
    CommandInvocation,
    ShellSession,
    module_prune_command,
)
from cli.module_cleaner import ModuleCleaner
from cli.snapshot_ledger import SnapshotLedger


def _write_module(path: Path, payload: object) -> None:
    if isinstance(payload, dict):
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    else:
        path.write_text(str(payload), encoding="utf-8")


def test_module_cleaner_analyze_detects_candidates(tmp_path: Path) -> None:
    modules_dir = tmp_path / "cli" / "modules"
    modules_dir.mkdir(parents=True, exist_ok=True)
    _write_module(
        modules_dir / "keep.json",
        {
            "name": "keep-module",
            "commands": [
                {"name": "hello", "summary": "Say hello", "response": "hi"},
            ],
        },
    )
    _write_module(
        modules_dir / "empty.json",
        {"name": "empty-module", "commands": []},
    )
    _write_module(
        modules_dir / "flagged.json",
        {
            "name": "deprecated-module",
            "retain": False,
            "commands": [{"name": "legacy", "response": "bye"}],
        },
    )
    _write_module(modules_dir / "invalid.json", "{" )  # invalid JSON payload

    ledger = SnapshotLedger(tmp_path / "cli" / "data" / "snapshot_ledger.jsonl")
    cleaner = ModuleCleaner(tmp_path, ledger)

    report = cleaner.analyze()

    assert report["counts"]["total"] == 4
    assert report["counts"]["removable"] == 3
    names = {entry["name"]: entry["action"] for entry in report["modules"]}
    assert names["keep-module"] == "keep"
    assert names["empty-module"] == "remove"
    assert names["deprecated-module"] == "remove"
    assert names["invalid"] == "remove"


def test_module_cleaner_prune_removes_files_and_records_event(tmp_path: Path) -> None:
    modules_dir = tmp_path / "cli" / "modules"
    modules_dir.mkdir(parents=True, exist_ok=True)
    _write_module(
        modules_dir / "keep.json",
        {
            "name": "keep-module",
            "commands": [{"name": "hello", "response": "hi"}],
        },
    )
    _write_module(
        modules_dir / "empty.json",
        {"name": "empty-module", "commands": []},
    )
    _write_module(modules_dir / "invalid.json", "{" )

    ledger = SnapshotLedger(tmp_path / "cli" / "data" / "snapshot_ledger.jsonl")
    cleaner = ModuleCleaner(tmp_path, ledger)

    result = cleaner.prune()

    assert not (modules_dir / "empty.json").exists()
    assert not (modules_dir / "invalid.json").exists()
    assert (modules_dir / "keep.json").exists()
    assert result["counts"]["removed"] == 2

    ledger_path = tmp_path / "cli" / "data" / "snapshot_ledger.jsonl"
    entries = [json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines()]
    event = next(entry for entry in reversed(entries) if entry.get("kind") == "module_entropy_pruned")
    removed_names = {item["name"] for item in event["removed"]}
    assert removed_names == {"empty-module", "invalid"}


def test_module_prune_command_supports_dry_run_and_removal(tmp_path: Path) -> None:
    modules_dir = tmp_path / "cli" / "modules"
    modules_dir.mkdir(parents=True, exist_ok=True)
    _write_module(
        modules_dir / "keep.json",
        {
            "name": "keep-module",
            "commands": [{"name": "hello", "response": "hi"}],
        },
    )
    _write_module(
        modules_dir / "remove.json",
        {"name": "remove-me", "commands": []},
    )

    shell = ShellSession(tmp_path)
    try:
        dry_run = module_prune_command(
            shell,
            CommandInvocation(name="module-prune", args=["--dry-run", "--json"]),
        )
        dry_payload = json.loads(dry_run.stdout)
        assert dry_payload["counts"]["removable"] == 1
        assert (modules_dir / "remove.json").exists()

        result = module_prune_command(
            shell,
            CommandInvocation(name="module-prune", args=["--json"]),
        )
    finally:
        shell.close()

    payload = json.loads(result.stdout)
    removed = {entry["name"] for entry in payload["removed"]}
    assert removed == {"remove-me"}
    assert not (modules_dir / "remove.json").exists()
    assert (modules_dir / "keep.json").exists()

