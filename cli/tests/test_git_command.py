from __future__ import annotations

import io
import subprocess
from pathlib import Path

from cli.command_shell import CommandInvocation, ShellSession, git_command


def _git(args: list[str], cwd: Path) -> None:
    subprocess.run(["git", *args], cwd=str(cwd), check=True, capture_output=True)


def _create_repository(root: Path) -> Path:
    origin = root / "origin"
    origin.mkdir()
    _git(["init"], origin)
    _git(["config", "user.email", "bot@example.com"], origin)
    _git(["config", "user.name", "Test Bot"], origin)
    (origin / "README.md").write_text("hello\n", encoding="utf-8")
    _git(["add", "README.md"], origin)
    _git(["commit", "-m", "init"], origin)
    return origin


def test_git_clone_local_repository(tmp_path: Path) -> None:
    origin = _create_repository(tmp_path)

    shell = ShellSession(tmp_path)
    try:
        result = git_command(
            shell,
            CommandInvocation(name="git", args=["clone", str(origin), "clone"]),
        )
    finally:
        shell.close()

    assert result.status == 0
    clone_path = tmp_path / "clone"
    assert clone_path.exists()
    assert (clone_path / "README.md").read_text(encoding="utf-8") == "hello\n"


def test_git_clone_rejects_path_escape(tmp_path: Path) -> None:
    origin = _create_repository(tmp_path)

    shell = ShellSession(tmp_path)
    try:
        result = git_command(
            shell,
            CommandInvocation(name="git", args=["clone", str(origin), "../outside"]),
        )
    finally:
        shell.close()

    assert result.status != 0
    assert "Permission denied" in result.stderr
    assert not (tmp_path.parent / "outside").exists()


def test_git_pass_through_command(tmp_path: Path) -> None:
    origin = _create_repository(tmp_path)

    shell = ShellSession(tmp_path)
    try:
        clone_result = git_command(
            shell,
            CommandInvocation(name="git", args=["clone", str(origin), "repo"]),
        )
        assert clone_result.status == 0
        shell.change_directory(Path("repo"))
        status = git_command(shell, CommandInvocation(name="git", args=["status", "--short"]))
    finally:
        shell.close()

    assert status.status == 0
    assert status.stdout.strip() == ""


def test_git_streams_output_in_interactive_session(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    shell = ShellSession(tmp_path)
    shell.stream_python_output = True

    class _StubProcess:
        def __init__(self) -> None:
            self.stdout = io.StringIO("hello")
            self.stderr = io.StringIO("warn")
            self.returncode = 0

        def wait(self, timeout: int | None = None) -> int:
            return self.returncode

        def poll(self) -> int:
            return self.returncode

        def send_signal(self, signum: int) -> None:  # pragma: no cover - not used
            self.returncode = 130

        def terminate(self) -> None:  # pragma: no cover - not used
            self.returncode = 143

        def kill(self) -> None:  # pragma: no cover - not used
            self.returncode = 137

    def _fake_popen(*args, **kwargs):
        return _StubProcess()

    monkeypatch.setattr(subprocess, "Popen", _fake_popen)

    try:
        invocation = CommandInvocation(name="git", args=["status"])
        result = git_command(shell, invocation)
    finally:
        shell.close()

    out, err = capsys.readouterr()
    assert out == "hello"
    assert err == "warn"
    assert result.streamed is True
    assert result.stdout == "hello\n"
    assert result.stderr == "warn\n"


def test_git_handles_keyboard_interrupt(tmp_path: Path, monkeypatch, capsys) -> None:
    shell = ShellSession(tmp_path)
    shell.stream_python_output = True

    class _InterruptProcess:
        def __init__(self) -> None:
            self.stdout = io.StringIO("partial")
            self.stderr = io.StringIO("")
            self.returncode: int | None = None
            self._interrupted = False

        def wait(self, timeout: int | None = None) -> int:
            if not self._interrupted:
                self._interrupted = True
                raise KeyboardInterrupt
            self.returncode = 130
            return self.returncode

        def poll(self) -> int | None:
            return self.returncode

        def send_signal(self, signum: int) -> None:
            self._interrupted = True
            self.returncode = 130

        def terminate(self) -> None:  # pragma: no cover - not used
            self.returncode = 143

        def kill(self) -> None:  # pragma: no cover - not used
            self.returncode = 137

    def _fake_popen(*args, **kwargs):
        return _InterruptProcess()

    monkeypatch.setattr(subprocess, "Popen", _fake_popen)

    try:
        invocation = CommandInvocation(name="git", args=["status"])
        result = git_command(shell, invocation)
    finally:
        shell.close()

    out, err = capsys.readouterr()
    assert out == "partial"
    assert err == ""
    assert result.streamed is True
    assert result.status == 130

