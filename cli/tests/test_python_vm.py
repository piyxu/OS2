from __future__ import annotations

import json
import types
from pathlib import Path

import pytest

from cli.command_shell import (
    CommandInvocation,
    ShellSession,
    pip_command,
    python_command,
    pyx_command,
)

from cli.import_verifier import compute_file_hash
from cli.python_vm import PythonVMLauncher, PythonVMError
from cli.snapshot_ledger import SnapshotLedger


def _read_ledger(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _read_kernel_log(root: Path) -> list[dict[str, object]]:
    log_path = root / "rust" / "os2-kernel" / "logs" / "kernel_events.jsonl"
    if not log_path.exists():
        return []
    return [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]


def _read_metadata(root: Path, sandbox_id: str) -> dict[str, object]:
    metadata_path = root / "cli" / "python_vm" / "sandboxes" / sandbox_id / "metadata.json"
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def _read_syspath_session(root: Path, session_id: str) -> dict[str, object]:
    payload_path = (
        root / "cli" / "python_vm" / "syspaths" / "sessions" / f"{session_id}.json"
    )
    return json.loads(payload_path.read_text(encoding="utf-8"))


def _read_snapshot_tag(root: Path, session_id: str) -> dict[str, object]:
    payload_path = (
        root / "cli" / "python_vm" / "snapshots" / "sessions" / f"{session_id}.json"
    )
    return json.loads(payload_path.read_text(encoding="utf-8"))


def test_launch_expression(tmp_path: Path) -> None:
    ledger_path = tmp_path / "ledger.jsonl"
    ledger = SnapshotLedger(ledger_path)
    launcher = PythonVMLauncher(tmp_path, ledger)

    result = launcher.launch(expr="1 + 2")

    assert result.exit_status == 0
    assert "3" in result.stdout
    assert result.tokens_consumed >= 1
    entries = _read_ledger(ledger_path)
    kinds = [entry["kind"] for entry in entries]
    assert kinds == [
        "python_vm_sandbox_created",
        "python_vm_start",
        "python_vm_async_queue_created",
        "python_vm_snapshot_tagged",
        "python_vm_syspath_synced",
        "python_vm_async_queue_drained",
        "python_vm_snapshot_state_saved",
        "python_vm_complete",
        "python_vm_sandbox_released",
    ]
    event_map = {entry["kind"]: entry for entry in entries}
    assert entries[-1]["status"] == "ok"
    assert event_map["python_vm_start"]["snapshot_id"] == result.snapshot_id
    assert event_map["python_vm_snapshot_tagged"]["snapshot_id"] == result.snapshot_id
    assert event_map["python_vm_syspath_synced"]["snapshot_id"] == result.snapshot_id
    assert event_map["python_vm_complete"]["snapshot_id"] == result.snapshot_id
    assert event_map["python_vm_complete"]["async_queue_status"] == "ok"
    assert event_map["python_vm_async_queue_drained"]["status"] == "ok"
    queue_summary = result.events["async_queue"]
    assert queue_summary["status"] == "ok"
    assert queue_summary["tasks_total"] == 0
    assert queue_summary["ledger_event_ids"]["created"] == event_map[
        "python_vm_async_queue_created"
    ]["event_id"]
    assert queue_summary["ledger_event_ids"]["drained"] == event_map[
        "python_vm_async_queue_drained"
    ]["event_id"]
    kernel_log = _read_kernel_log(tmp_path)
    assert kernel_log[-1]["detail"]["command_alias"] is None
    kernel_events = result.events["kernel_log"]
    assert kernel_events["stdout"]["chain_hash"] == kernel_log[0]["chain_hash"]
    syspath_event = event_map["python_vm_syspath_synced"]
    assert syspath_event["count"] == len(syspath_event["paths"])
    metadata = _read_metadata(tmp_path, result.sandbox_id)
    assert metadata["tokens_consumed"] == result.tokens_consumed
    assert metadata["status"] == "ok"
    assert metadata["sys_path_event_id"] == syspath_event["event_id"]
    assert metadata["sys_path_hash"] == syspath_event["paths_hash"]
    assert metadata["sys_path"]
    assert metadata["kernel_log_path"].endswith("kernel_events.jsonl")
    assert metadata["snapshot_id"] == result.snapshot_id
    assert metadata["snapshot_event_id"] == event_map["python_vm_snapshot_tagged"]["event_id"]
    assert metadata["async_queue"]["status"] == "ok"
    assert metadata["async_queue"]["tasks_total"] == 0
    assert [entry["label"] for entry in kernel_log] == [
        "python_vm_stream",
        "python_vm_stream",
        "python_vm_session",
    ]
    stdout_event = kernel_log[0]
    assert stdout_event["detail"]["stream"] == "stdout"
    assert "3" in stdout_event["detail"]["content"]
    stderr_event = kernel_log[1]
    assert stderr_event["detail"]["stream"] == "stderr"
    assert stderr_event["detail"]["is_empty"] is True
    session_event = kernel_log[2]
    assert session_event["detail"]["status"] == "ok"
    assert session_event["detail"]["command_alias"] is None
    assert session_event["detail"]["stdout_event_chain_hash"] == stdout_event["chain_hash"]
    assert session_event["detail"]["stderr_event_chain_hash"] == stderr_event["chain_hash"]
    assert stdout_event["detail"]["snapshot_id"] == result.snapshot_id
    assert session_event["detail"]["snapshot_id"] == result.snapshot_id
    assert metadata["kernel_log_events"]["stdout"] == stdout_event["chain_hash"]
    assert metadata["kernel_log_path"].endswith("kernel_events.jsonl")
    session_payload = _read_syspath_session(tmp_path, result.session_id)
    assert session_payload["paths_hash"] == syspath_event["paths_hash"]
    assert session_payload["session_id"] == result.session_id
    assert session_payload["snapshot_id"] == result.snapshot_id
    snapshot_payload = _read_snapshot_tag(tmp_path, result.session_id)
    assert snapshot_payload["snapshot_id"] == result.snapshot_id
    assert snapshot_payload["event_id"] == event_map["python_vm_snapshot_tagged"]["event_id"]
    assert snapshot_payload["ledger_event_ids"]["start"] == event_map["python_vm_start"]["event_id"]


def test_launch_script(tmp_path: Path) -> None:
    ledger_path = tmp_path / "ledger.jsonl"
    ledger = SnapshotLedger(ledger_path)
    launcher = PythonVMLauncher(tmp_path, ledger)

    script = tmp_path / "cli" / "python_vm" / "example.py"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text("print('hello')\n", encoding="utf-8")

    result = launcher.launch(script_path=script)

    assert result.stdout.strip() == "hello"
    entries = _read_ledger(ledger_path)
    assert entries[-1]["status"] == "ok"
    assert any(entry["kind"] == "python_vm_syspath_synced" for entry in entries)
    assert any(entry["kind"] == "python_vm_snapshot_tagged" for entry in entries)
    queue_summary = result.events["async_queue"]
    assert queue_summary["status"] == "ok"
    metadata = _read_metadata(tmp_path, result.sandbox_id)
    assert Path(metadata["script"]).name == "example.py"
    assert metadata["status"] == "ok"
    assert metadata["async_queue"]["status"] == "ok"
    kernel_log = _read_kernel_log(tmp_path)
    assert kernel_log[-1]["detail"]["status"] == "ok"
    assert "hello" in kernel_log[0]["detail"]["content"]


def test_rejects_out_of_root(tmp_path: Path) -> None:
    ledger_path = tmp_path / "ledger.jsonl"
    ledger = SnapshotLedger(ledger_path)
    launcher = PythonVMLauncher(tmp_path, ledger)

    with pytest.raises(PythonVMError):
        launcher.launch(script_path=Path("/etc/passwd"))


def test_python_alias_routes_into_vm(tmp_path: Path) -> None:
    shell = ShellSession(tmp_path)
    try:
        result = python_command(shell, CommandInvocation(name="python", args=["-c", "1 + 2"]))
    finally:
        shell.close()

    assert result.status == 0
    assert "3" in result.stdout
    ledger_path = tmp_path / "cli" / "data" / "snapshot_ledger.jsonl"
    entries = _read_ledger(ledger_path)
    kinds = [entry["kind"] for entry in entries]
    expected_prefix = [
        "python_vm_sandbox_created",
        "python_vm_start",
        "python_vm_async_queue_created",
        "python_vm_snapshot_tagged",
        "python_vm_syspath_synced",
        "python_vm_async_queue_drained",
        "python_vm_snapshot_state_saved",
        "python_vm_complete",
    ]
    assert kinds[: len(expected_prefix)] == expected_prefix
    assert kinds[len(expected_prefix)] == "python_vm_sandbox_released"
    assert kinds[-1] == "python_state_merged"
    assert entries[len(expected_prefix)]["status"] == "ok"


def test_python_command_accepts_token_budget_override(tmp_path: Path) -> None:
    shell = ShellSession(tmp_path)
    try:
        result = python_command(
            shell,
            CommandInvocation(
                name="python",
                args=["--token-budget", "512", "-c", "print('override')"],
            ),
        )
    finally:
        shell.close()

    assert result.status == 0
    assert "override" in result.stdout
    assert result.audit["token_budget"] == 512


def test_python_command_rejects_non_positive_token_budget(tmp_path: Path) -> None:
    shell = ShellSession(tmp_path)
    try:
        result = python_command(
            shell,
            CommandInvocation(
                name="python",
                args=["--token-budget", "0", "-c", "print('fail')"],
            ),
        )
    finally:
        shell.close()

    assert result.status == 1
    assert "--token-budget expects a positive integer" in result.stderr


def test_shell_session_honors_env_script_token_budget(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("OS2_SCRIPT_TOKEN_BUDGET", "4096")
    shell = ShellSession(tmp_path)
    try:
        assert shell.config["script_token_budget"] == 4096
    finally:
        shell.close()
    monkeypatch.delenv("OS2_SCRIPT_TOKEN_BUDGET", raising=False)


def test_python_vm_exposes_standard_builtins(tmp_path: Path) -> None:
    ledger_path = tmp_path / "ledger.jsonl"
    ledger = SnapshotLedger(ledger_path)
    launcher = PythonVMLauncher(tmp_path, ledger)

    payload_path = tmp_path / "cli" / "python_vm" / "payload.txt"
    payload_path.parent.mkdir(parents=True, exist_ok=True)

    script = tmp_path / "cli" / "python_vm" / "check_builtins.py"
    script.write_text(
        "import builtins\n"
        "import pathlib\n"
        f"target = pathlib.Path(r\"{payload_path}\")\n"
        "with builtins.open(target, 'w', encoding='utf-8') as handle:\n"
        "    handle.write('data')\n"
        "print(callable(builtins.input))\n",
        encoding="utf-8",
    )

    result = launcher.launch(script_path=script)

    assert result.exit_status == 0
    assert "True" in result.stdout
    assert payload_path.read_text(encoding="utf-8") == "data"


def test_python_vm_handles_system_exit(tmp_path: Path) -> None:
    ledger_path = tmp_path / "ledger.jsonl"
    ledger = SnapshotLedger(ledger_path)
    launcher = PythonVMLauncher(tmp_path, ledger)

    script_ok = tmp_path / "cli" / "python_vm" / "exit_ok.py"
    script_ok.parent.mkdir(parents=True, exist_ok=True)
    script_ok.write_text("import sys\nsys.exit(0)\n", encoding="utf-8")

    result_ok = launcher.launch(script_path=script_ok)
    assert result_ok.exit_status == 0
    assert result_ok.stderr.strip() == ""

    script_error = tmp_path / "cli" / "python_vm" / "exit_error.py"
    script_error.write_text("import sys\nsys.exit('failure')\n", encoding="utf-8")

    result_error = launcher.launch(script_path=script_error)
    assert result_error.exit_status == 1
    assert "failure" in result_error.stderr


def test_pip_command_handles_system_exit(tmp_path: Path) -> None:
    shell = ShellSession(tmp_path)
    pip_pkg = tmp_path / "pip"
    pip_pkg.mkdir()
    (pip_pkg / "__init__.py").write_text(
        "import sys\n"
        "print('pip shim start')\n"
        "sys.exit(0)\n",
        encoding="utf-8",
    )

    try:
        result = pip_command(shell, CommandInvocation(name="pip", args=[]))
    finally:
        shell.close()

    assert result.status == 0
    assert "pip shim start" in result.stdout


def test_python_command_executes_script(tmp_path: Path) -> None:
    script = tmp_path / "sample.py"
    script.write_text("print('hello from script')\n", encoding="utf-8")

    shell = ShellSession(tmp_path)
    try:
        result = python_command(shell, CommandInvocation(name="python", args=["sample.py"]))
    finally:
        shell.close()

    assert result.status == 0
    assert "hello from script" in result.stdout
    assert result.audit.get("script") == str(script)


def test_python_command_streams_output_when_interactive(tmp_path: Path, capsys) -> None:
    script = tmp_path / "interactive.py"
    script.write_text("print('ready for input')\n", encoding="utf-8")

    shell = ShellSession(tmp_path)
    shell.stream_python_output = True
    try:
        result = python_command(
            shell, CommandInvocation(name="python", args=[script.name])
        )
    finally:
        shell.close()

    captured = capsys.readouterr()
    assert "ready for input" in captured.out
    summary_fragment = f"[python:{result.audit['session_id']}@{result.audit['sandbox_id']}]"
    assert summary_fragment in captured.out
    assert result.streamed is True


def test_python_command_reports_tty_when_streaming(tmp_path: Path, capsys) -> None:
    script = tmp_path / "isatty.py"
    script.write_text("import sys\nprint(sys.stdout.isatty())\n", encoding="utf-8")

    shell = ShellSession(tmp_path)
    shell.stream_python_output = True
    try:
        result = python_command(shell, CommandInvocation(name="python", args=[script.name]))
    finally:
        shell.close()

    captured = capsys.readouterr()
    assert "True" in captured.out
    assert result.streamed is True


def test_python_command_handles_legacy_vm_signature(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    script = tmp_path / "legacy.py"
    script.write_text("print('legacy support works')\n", encoding="utf-8")

    shell = ShellSession(tmp_path)
    legacy_launch = shell.python_vm.launch

    def _legacy_launch(**kwargs):
        if "stream_output" in kwargs:
            raise TypeError("launch() got an unexpected keyword argument 'stream_output'")
        base_result = legacy_launch(**kwargs)
        payload = {k: v for k, v in base_result.__dict__.items() if k != "streamed"}
        return types.SimpleNamespace(**payload)

    monkeypatch.setattr(shell.python_vm, "launch", _legacy_launch, raising=False)
    shell.stream_python_output = True
    try:
        result = python_command(shell, CommandInvocation(name="python", args=[script.name]))
    finally:
        shell.close()

    assert result.status == 0
    assert "legacy support works" in result.stdout
    assert result.streamed is False


def test_python_module_execution(tmp_path: Path) -> None:
    module_path = tmp_path / "demo_module.py"
    module_path.write_text("print('module works')\n", encoding="utf-8")

    shell = ShellSession(tmp_path)
    try:
        result = python_command(
            shell, CommandInvocation(name="python", args=["-m", "demo_module"])
        )
    finally:
        shell.close()

    assert result.status == 0
    assert "module works" in result.stdout
    assert result.audit.get("module") == "demo_module"


def test_python_version_flag(tmp_path: Path) -> None:
    shell = ShellSession(tmp_path)
    try:
        result = python_command(shell, CommandInvocation(name="python", args=["--version"]))
    finally:
        shell.close()

    assert result.status == 0
    first_line = result.stdout.splitlines()[0]
    assert first_line.startswith("Python ")


def test_pip_command_invokes_module(tmp_path: Path) -> None:
    pip_pkg = tmp_path / "pip"
    pip_pkg.mkdir()
    (pip_pkg / "__init__.py").write_text("__all__ = ['__main__']\n", encoding="utf-8")
    (pip_pkg / "__main__.py").write_text(
        "import sys\nprint('pip invoked:' + ' '.join(sys.argv[1:]))\n",
        encoding="utf-8",
    )

    shell = ShellSession(tmp_path)
    try:
        result = pip_command(shell, CommandInvocation(name="pip", args=["--help"]))
    finally:
        shell.close()

    assert result.status == 0
    first_line = result.stdout.splitlines()[0]
    assert "pip invoked:" in first_line
    assert "--help" in first_line


def test_pyx_alias_executes_scripts(tmp_path: Path) -> None:
    script = tmp_path / "demo.py"
    script.write_text("print('alias works')\n", encoding="utf-8")

    shell = ShellSession(tmp_path)
    try:
        result = pyx_command(shell, CommandInvocation(name="pyx", args=["demo.py"]))
    finally:
        shell.close()

    assert result.status == 0
    assert "alias works" in result.stdout
    kernel_log = _read_kernel_log(tmp_path)
    assert kernel_log[-1]["detail"]["command_alias"] == "pyx"


def test_unique_sandboxes_per_session(tmp_path: Path) -> None:
    ledger_path = tmp_path / "ledger.jsonl"
    ledger = SnapshotLedger(ledger_path)
    launcher = PythonVMLauncher(tmp_path, ledger)

    first = launcher.launch(expr="1")
    second = launcher.launch(expr="2")

    assert first.sandbox_id != second.sandbox_id
    meta_first = _read_metadata(tmp_path, first.sandbox_id)
    meta_second = _read_metadata(tmp_path, second.sandbox_id)
    assert meta_first["tokens_consumed"] >= 1
    assert meta_second["tokens_consumed"] >= 1
    assert meta_first["status"] == "ok"
    assert meta_second["status"] == "ok"


def test_async_queue_executes_tasks(tmp_path: Path) -> None:
    ledger_path = tmp_path / "ledger.jsonl"
    ledger = SnapshotLedger(ledger_path)
    launcher = PythonVMLauncher(tmp_path, ledger)

    script = tmp_path / "cli" / "python_vm" / "async_tasks.py"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(
        "\n".join(
            [
                "import asyncio",
                "",
                "async def produce(name, delay=0):",
                "    await asyncio.sleep(delay)",
                "    return name.upper()",
                "",
                "async_queue.schedule('first', produce('one'))",
                "async_queue.schedule('second', lambda: produce('two'))",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = launcher.launch(script_path=script)

    assert result.exit_status == 0
    queue_summary = result.events["async_queue"]
    assert queue_summary["status"] == "ok"
    assert queue_summary["tasks_total"] == 2
    assert queue_summary["tasks_error"] == 0
    task_names = [task["name"] for task in queue_summary["tasks"]]
    assert task_names == ["first", "second"]
    metadata = _read_metadata(tmp_path, result.sandbox_id)
    assert metadata["async_queue"]["tasks_total"] == 2
    assert metadata["async_queue"]["tasks_error"] == 0
    entries = _read_ledger(ledger_path)
    kinds = [entry["kind"] for entry in entries]
    assert "python_vm_async_task_queued" in kinds
    assert "python_vm_async_task_started" in kinds
    assert "python_vm_async_task_completed" in kinds


def test_async_queue_failure_sets_error(tmp_path: Path) -> None:
    ledger_path = tmp_path / "ledger.jsonl"
    ledger = SnapshotLedger(ledger_path)
    launcher = PythonVMLauncher(tmp_path, ledger)

    script = tmp_path / "cli" / "python_vm" / "async_failure.py"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(
        "\n".join(
            [
                "async def boom():",
                "    raise RuntimeError('boom')",
                "",
                "async_queue.schedule('boom', boom)",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = launcher.launch(script_path=script)

    assert result.exit_status == 1
    assert "[async:boom] RuntimeError: boom" in result.stderr
    queue_summary = result.events["async_queue"]
    assert queue_summary["status"] == "error"
    assert queue_summary["tasks_error"] == 1
    metadata = _read_metadata(tmp_path, result.sandbox_id)
    assert metadata["async_queue"]["status"] == "error"
    assert metadata["async_queue"]["tasks_error"] == 1
    entries = _read_ledger(ledger_path)
    event_map = {entry["kind"]: entry for entry in entries}
    assert event_map["python_vm_complete"]["status"] == "error"
    assert event_map["python_vm_complete"]["async_queue_status"] == "error"


def test_budget_exceeded_records_release(tmp_path: Path) -> None:
    ledger_path = tmp_path / "ledger.jsonl"
    ledger = SnapshotLedger(ledger_path)
    launcher = PythonVMLauncher(tmp_path, ledger, default_token_budget=4)

    with pytest.raises(PythonVMError):
        launcher.launch(expr=" " * 200)

    entries = _read_ledger(ledger_path)
    assert [entry["kind"] for entry in entries] == [
        "python_vm_sandbox_created",
        "python_vm_sandbox_released",
    ]
    assert entries[-1]["status"] == "budget_exceeded"


def _write_manifest(root: Path, manifest: dict[str, dict[str, str]]) -> Path:
    manifest_path = root / "cli" / "python_vm" / "import_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def test_import_verification_allows_manifest_module(tmp_path: Path) -> None:
    module = tmp_path / "trusted_module.py"
    module.write_text("VALUE = 7\n", encoding="utf-8")

    manifest = {
        "trusted_module": {
            "path": str(module.relative_to(tmp_path)),
            "hash": compute_file_hash(module),
        }
    }
    _write_manifest(tmp_path, manifest)

    ledger = SnapshotLedger(tmp_path / "ledger.jsonl")
    launcher = PythonVMLauncher(tmp_path, ledger)

    result = launcher.launch(expr="__import__('trusted_module').VALUE")

    assert result.exit_status == 0
    entries = _read_ledger(tmp_path / "ledger.jsonl")
    kinds = [entry["kind"] for entry in entries]
    assert "python_vm_import_verified" in kinds
    assert "python_vm_snapshot_tagged" in kinds
    metadata = _read_metadata(tmp_path, result.sandbox_id)
    imports = metadata.get("imports", [])
    assert any(entry["module"] == "trusted_module" and entry["status"] == "verified" for entry in imports)


def test_import_verification_blocks_hash_mismatch(tmp_path: Path) -> None:
    module = tmp_path / "trusted_module.py"
    module.write_text("VALUE = 1\n", encoding="utf-8")

    manifest = {
        "trusted_module": {
            "path": str(module.relative_to(tmp_path)),
            "hash": compute_file_hash(module),
        }
    }
    _write_manifest(tmp_path, manifest)

    # Modify the module after manifest creation so the hash no longer matches.
    module.write_text("VALUE = 2\n", encoding="utf-8")

    ledger = SnapshotLedger(tmp_path / "ledger.jsonl")
    launcher = PythonVMLauncher(tmp_path, ledger)

    result = launcher.launch(expr="__import__('trusted_module')")

    assert result.exit_status == 1
    ledger_entries = _read_ledger(tmp_path / "ledger.jsonl")
    kinds = [entry["kind"] for entry in ledger_entries]
    assert "python_vm_import_blocked" in kinds
    assert "python_vm_snapshot_tagged" in kinds
    metadata = _read_metadata(tmp_path, result.sandbox_id)
    imports = metadata.get("imports", [])
    assert any(entry["status"] == "blocked" for entry in imports)
    kernel_log = _read_kernel_log(tmp_path)
    assert kernel_log[-1]["detail"]["status"] == "error"
    assert not kernel_log[1]["detail"]["is_empty"]
