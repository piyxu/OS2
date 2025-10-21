import json
from pathlib import Path

from cli.command_shell import (
    CommandInvocation,
    ShellSession,
    task_proposals_command,
)


def _load_json(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def test_task_proposals_registers_roken_entry(tmp_path: Path) -> None:
    shell = ShellSession(tmp_path)
    try:
        result = task_proposals_command(
            shell,
            CommandInvocation(
                name="task-proposals",
                args=[
                    "--json",
                    "propose",
                    "roken-assembly",
                    "Enable deterministic proposal staging",
                    "--description",
                    "Draft roadmap tasks from replay metrics",
                    "--tag",
                    "phase-4",
                    "--tag",
                    "analysis",
                ],
            ),
        )
        listing = task_proposals_command(
            shell,
            CommandInvocation(name="task-proposals", args=["--json", "list"]),
        )
    finally:
        shell.close()

    payload = json.loads(result.stdout)
    proposal = payload["proposal"]
    assert proposal["proposal_id"] == 1
    assert proposal["source"] == "roken-assembly"
    assert sorted(proposal["tags"]) == ["analysis", "phase-4"]

    listing_payload = json.loads(listing.stdout)
    assert listing_payload["proposals"][0]["title"] == "Enable deterministic proposal staging"

    data_path = tmp_path / "cli" / "data" / "task_proposals.json"
    stored = _load_json(data_path)
    assert stored[0]["source"] == "roken-assembly"

    ledger_path = tmp_path / "cli" / "data" / "snapshot_ledger.jsonl"
    lines = ledger_path.read_text(encoding="utf-8").splitlines()
    parsed = (json.loads(line) for line in reversed(lines))
    entry = next(item for item in parsed if item.get("kind") == "task_proposal_registered")
    assert entry["proposal"]["proposal_id"] == 1


def test_task_proposals_rejects_unknown_source(tmp_path: Path) -> None:
    shell = ShellSession(tmp_path)
    try:
        result = task_proposals_command(
            shell,
            CommandInvocation(
                name="task-proposals",
                args=["propose", "mystery", "Investigate"],
            ),
        )
    finally:
        shell.close()

    assert result.status == 1
    assert "Unknown proposal source" in result.stderr
