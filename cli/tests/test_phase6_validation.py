import json
from pathlib import Path

from cli.command_shell import (
    CommandInvocation,
    ShellSession,
    deterministic_benchmark_command,
    python_command,
    uname,
)
from cli.signature import SignatureVerifier


def _read_ledger(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_phase6_validation_suite(tmp_path: Path) -> None:
    shell = ShellSession(tmp_path)
    try:
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir(parents=True, exist_ok=True)
        (docs_dir / "yol_hikayesi.md").write_text(
            "\n".join(
                [
                    "## Phase 6 — Deterministic Validation",
                    "- [x] F6.1 verify deterministic python",
                    "- [x] F6.2 stress kernel",
                    "- [x] F6.3 replay 1000 snapshots",
                    "- [x] F6.4 update ledger",
                    "- [x] F6.5 export hash-signed tasks",
                    "- [x] F6.6 consolidate benchmark",
                    "- [x] F6.7 summarize via cli",
                    "- [x] F6.8 render ai graph",
                    "- [x] F6.9 verify snapshots",
                    "- [x] F6.10 produce build signature",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        python_run = python_command(
            shell, CommandInvocation(name="python", args=["-c", "print('phase6')"])
        )
        assert python_run.status == 0

        run_result = deterministic_benchmark_command(
            shell,
            CommandInvocation(
                name="deterministic-benchmark",
                args=["--json", "run", "--count", "4", "--stress-iterations", "5"],
            ),
        )
        assert run_result.status == 0
        run_payload = json.loads(run_result.stdout)

        python_summary = run_payload["python_sessions"]
        assert python_summary["checked_sessions"] >= 1
        assert python_summary["checked_sessions"] == python_summary["verified_sessions"]
        assert python_summary["failures"] == []

        stress_summary = run_payload["stress"]
        assert stress_summary["iterations"] == 5
        assert len(stress_summary["metrics"]) == 5

        replay_summary = run_payload["replay_consistency"]
        assert replay_summary["count"] == 4
        assert len(replay_summary["sequence_preview"]) == 4

        ai_report = run_payload["ai_report"]
        assert "▇" in ai_report["graph"]
        assert len(ai_report["tests"]) == len(stress_summary["metrics"])

        snapshot_summary = run_payload["snapshot_verification"]
        component_digests = snapshot_summary["digests"]
        assert set(component_digests) == {"python", "stress", "replay"}
        assert len(snapshot_summary["combined_digest"]) == 64

        export_summary = run_payload["journey_export"]
        export_path = Path(export_summary["path"])
        assert export_path.exists()

        export_payload = json.loads(export_path.read_text(encoding="utf-8"))
        assert export_payload["digest"] == export_summary["digest"]
        signer = SignatureVerifier({"default": "os2-model-signing"}, default_key="default")
        signer.verify_digest(export_payload["digest"], export_payload["signature"])
        assert len(export_payload["payload"]["tasks"]) == 10

        build_summary = run_payload["build_signature"]
        build_path = Path(build_summary["path"])
        assert build_path.exists()
        build_payload = json.loads(build_path.read_text(encoding="utf-8"))
        assert build_payload["signature"] == build_summary["signature"]
        signer.verify_digest(build_payload["digest"], build_payload["signature"])

        state_path = tmp_path / "cli" / "data" / "deterministic_benchmark.json"
        state_payload = json.loads(state_path.read_text(encoding="utf-8"))
        assert state_payload["last_run"]["run_id"] == run_payload["run_id"]
        assert state_payload["next_run_id"] == run_payload["run_id"] + 1

        status_text = deterministic_benchmark_command(
            shell, CommandInvocation(name="deterministic-benchmark", args=["status"])
        )
        assert status_text.status == 0
        assert "Deterministic benchmark run" in status_text.stdout
        assert "AI graph" in status_text.stdout

        status_json = deterministic_benchmark_command(
            shell, CommandInvocation(name="deterministic-benchmark", args=["--json", "status"])
        )
        assert status_json.status == 0
        status_payload = json.loads(status_json.stdout)
        assert status_payload["run_id"] == run_payload["run_id"]
        assert status_payload["ai_report"] == ai_report

        uname_result = uname(shell, CommandInvocation(name="uname", args=[]))
        assert uname_result.stdout.strip() == "Piyxu OS2 0.1.0v Kernel (deterministic)"
    finally:
        shell.close()

    ledger_path = tmp_path / "cli" / "data" / "snapshot_ledger.jsonl"
    ledger_entries = _read_ledger(ledger_path)
    kinds = {entry["kind"] for entry in ledger_entries}
    expected_kinds = {
        "python_vm_replay_verified",
        "kernel_ai_stress_test_completed",
        "snapshot_replay_consistency_measured",
        "journey_exported",
        "deterministic_snapshot_hash_verified",
        "piyxu_build_signature_produced",
        "deterministic_benchmark_suite_completed",
    }
    assert expected_kinds.issubset(kinds)
