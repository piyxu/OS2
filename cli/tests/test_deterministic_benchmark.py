import json
from pathlib import Path

from cli.command_shell import (
    CommandInvocation,
    ShellSession,
    deterministic_benchmark_command,
    python_command,
)


def _read_ledger(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_deterministic_benchmark_run_and_status(tmp_path: Path) -> None:
    shell = ShellSession(tmp_path)
    try:
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir(parents=True, exist_ok=True)
        (docs_dir / "yol_hikayesi.md").write_text(
            "\n".join(
                [
                    "## Phase 6 — Deterministic Validation",
                    "- [x] F6.1 sample",
                    "- [x] F6.2 sample",
                    "- [x] F6.3 sample",
                    "- [x] F6.4 sample",
                    "- [x] F6.5 sample",
                    "- [x] F6.6 sample",
                    "- [x] F6.7 sample",
                    "- [x] F6.8 sample",
                    "- [x] F6.9 sample",
                    "- [x] F6.10 sample",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        python_result = python_command(
            shell,
            CommandInvocation(name="python", args=["-c", "print('suite')"]),
        )
        assert python_result.status == 0

        run_result = deterministic_benchmark_command(
            shell,
            CommandInvocation(
                name="deterministic-benchmark",
                args=[
                    "--json",
                    "run",
                    "--count",
                    "5",
                    "--stress-iterations",
                    "3",
                ],
            ),
        )
        assert run_result.status == 0
        run_payload = json.loads(run_result.stdout)
        assert run_payload["python_sessions"]["verified_sessions"] >= 1
        assert run_payload["replay_consistency"]["count"] == 5
        assert run_payload["stress"]["iterations"] == 3
        export_path = Path(run_payload["journey_export"]["path"])
        assert export_path.exists()
        assert run_payload["build_signature"]["signature"]

        status_result = deterministic_benchmark_command(
            shell,
            CommandInvocation(
                name="deterministic-benchmark",
                args=["--json", "status"],
            ),
        )
        assert status_result.status == 0
        status_payload = json.loads(status_result.stdout)
        assert status_payload["run_id"] == run_payload["run_id"]
    finally:
        shell.close()

    ledger_path = tmp_path / "cli" / "data" / "snapshot_ledger.jsonl"
    kinds = [entry["kind"] for entry in _read_ledger(ledger_path)]
    assert "deterministic_benchmark_suite_completed" in kinds
    assert "journey_exported" in kinds
    assert "piyxu_build_signature_produced" in kinds
