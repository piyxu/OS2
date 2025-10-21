#!/usr/bin/env python3
"""Interactive Piyxu OS2 0.1.0v command shell implementing the command task list."""

from __future__ import annotations

import argparse
import dataclasses
import datetime as _dt
import functools
import hashlib
import io
import json
import logging
import os
import re
import readline
import shlex
import signal
import subprocess
import sys
import threading
import time
import urllib.parse
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from cli.ai_executor import DeterministicAIExecutor
from cli.ai_replay import AIReplayError, AIReplayManager
from cli.boot_model_loader import BootModelLoader
from cli.cas import ContentAddressableStore
from cli.deterministic_benchmark import (
    DeterministicBenchmarkError,
    DeterministicBenchmarkRunner,
)
from cli.documentation import DocumentationPublisher, DocumentationError
from cli.deterministic_recompile import (
    DeterministicRecompileError,
    DeterministicRecompileManager,
)
from cli.download_manager import (
    DownloadError,
    SecureDownloadManager,
    TokenBudgetExceeded,
    TokenBudgetLedger,
)
from cli.feedback_hooks import register_default_feedback_hooks
from cli.ledger_inspect import LedgerInspector
from cli.hf_client import HuggingFaceAPIError, SignedHuggingFaceClient
from cli.gpu_manager import GPUAccessManager, GPUAccessError
from cli.kernel_log import KernelLogWriter
from cli.kernel_metrics import KernelTaskAnalyzer
from cli.kernel_performance import (
    KernelPerformanceError,
    KernelPerformanceMonitor,
)
from cli.kernel_updates import KernelUpdateDistributor, KernelUpdateError
from cli.llm_adapter import DeterministicLLMAdapter
from cli.model_registry import ModelRecord, ModelRegistry, ModelRegistryError
from cli.model_sources import ModelSourceError, load_manifest
from cli.model_verifier import ModelArtifactVerificationError
from cli.module_cleaner import ModuleCleaner, ModuleCleanerError
from cli.living_system import (
    LivingDeterministicSystemError,
    LivingDeterministicSystemManager,
)
from cli.signature import SignatureVerifier, SignatureError
from cli.snapshot_benchmark import (
    SnapshotBenchmarkError,
    SnapshotBenchmarkManager,
)
from cli.snapshot_ledger import SnapshotLedger, SnapshotLedgerError
from cli.runtime_loader import RuntimeLoader, RuntimeLoaderError
from cli.python_env import (
    InvalidPythonEnvironmentName,
    PythonEnvironmentError,
    PythonEnvironmentExistsError,
    PythonEnvironmentManager,
)
from cli.python_vm import PythonVMLauncher, PythonVMError
from cli.python_determinism import PythonDeterminismError, PythonDeterminismVerifier
from cli.self_feedback import SelfFeedbackAnalyzer, SelfFeedbackError
from cli.self_task_review import SelfTaskReviewError, SelfTaskReviewModule
from cli.task_proposals import TaskProposalError, TaskProposalRegistry
from cli.time_travel import TimeTravelDebugger
from cli.module_permissions import ModulePermissionRegistry, ModulePermissionError
from cli.audit_diff import LedgerAuditDiff
from cli.session_auth import SnapshotAuthenticator, SessionAuthenticationError
from cli.entropy_audit import EntropyAuditor
from cli.integrity_monitor import IntegrityMonitor
from cli.backup_manager import SecureBackupManager
from cli.security_log import SecurityLogManager

OS_NAME = "Piyxu OS2 0.1.0v"

# ---------------------------------------------------------------------------
# Localization helpers
# ---------------------------------------------------------------------------


def now_utc() -> _dt.datetime:
    return _dt.datetime.now(_dt.timezone.utc)


def isoformat_utc(dt: _dt.datetime) -> str:
    return dt.isoformat().replace("+00:00", "Z")


@dataclass
class LocalizedString:
    """Provides localized content with a default fallback."""

    default: str
    translations: Dict[str, str] = field(default_factory=dict)

    def resolve(self, locale: str) -> str:
        return self.translations.get(locale, self.default)


# ---------------------------------------------------------------------------
# Command invocation/result types
# ---------------------------------------------------------------------------


@dataclass
class CommandInvocation:
    name: str
    args: List[str]
    stdin: Optional[str] = None
    pipeline_index: int = 0
    pipeline_length: int = 1


@dataclass
class CommandResult:
    stdout: str = ""
    stderr: str = ""
    status: int = 0
    audit: Dict[str, Any] = field(default_factory=dict)
    streamed: bool = False

    def merge(self, other: "CommandResult") -> None:
        self.stdout += other.stdout
        self.stderr += other.stderr
        self.status = other.status
        self.audit.update(other.audit)
        self.streamed = self.streamed or other.streamed


# ---------------------------------------------------------------------------
# Capability management, rate limiting, and auditing
# ---------------------------------------------------------------------------


_GIT_CLONE_VALUE_OPTIONS = {
    "-b",
    "--branch",
    "-c",
    "--config",
    "-o",
    "--origin",
    "--reference",
    "--separate-git-dir",
    "--depth",
    "--shallow-since",
    "--shallow-exclude",
    "--server-option",
    "--upload-pack",
    "--template",
    "--filter",
    "--jobs",
    "--dissociate",
}


@dataclass
class RateLimit:
    limit: int
    per_seconds: int


class RateLimiter:
    def __init__(self) -> None:
        self._invocations: Dict[str, deque[float]] = defaultdict(deque)

    def check(self, key: str, limit: RateLimit) -> bool:
        window = self._invocations[key]
        now = time.time()
        cutoff = now - limit.per_seconds
        while window and window[0] < cutoff:
            window.popleft()
        if len(window) >= limit.limit:
            return False
        window.append(now)
        return True


# ---------------------------------------------------------------------------
# Process and job management
# ---------------------------------------------------------------------------


@dataclass
class Process:
    pid: int
    name: str
    user: str
    cpu_percent: float
    memory_kb: int
    state: str = "Running"
    started_at: _dt.datetime = field(default_factory=now_utc)


@dataclass
class Job:
    job_id: int
    command_line: str
    thread: Optional[threading.Thread]
    status: str = "running"
    result: Optional[CommandResult] = None


class ProcessTable:
    def __init__(self, owner: str) -> None:
        self._owner = owner
        self._processes: Dict[int, Process] = {}
        self._next_pid = 1000
        self._seed_default_processes()

    def _seed_default_processes(self) -> None:
        for name, cpu, mem in (
            ("shell", 0.5, 128),
            ("kernel-scheduler", 12.0, 4096),
            ("event-daemon", 4.5, 2048),
            ("memory-manager", 6.1, 8192),
        ):
            self.spawn(name=name, cpu_percent=cpu, memory_kb=mem, state="Running")

    def spawn(
        self,
        name: str,
        cpu_percent: float,
        memory_kb: int,
        state: str = "Sleeping",
    ) -> Process:
        pid = self._next_pid
        self._next_pid += 1
        proc = Process(pid=pid, name=name, user=self._owner, cpu_percent=cpu_percent, memory_kb=memory_kb, state=state)
        self._processes[pid] = proc
        return proc

    def all(self) -> List[Process]:
        return sorted(self._processes.values(), key=lambda p: p.pid)

    def get(self, pid: int) -> Optional[Process]:
        return self._processes.get(pid)

    def kill(self, pid: int) -> bool:
        return self._processes.pop(pid, None) is not None

    def update_state(self, pid: int, state: str) -> bool:
        proc = self._processes.get(pid)
        if not proc:
            return False
        proc.state = state
        return True


# ---------------------------------------------------------------------------
# Shell session and command registry
# ---------------------------------------------------------------------------


Handler = Callable[["ShellSession", CommandInvocation], CommandResult]


@dataclass
class Command:
    name: str
    summary: LocalizedString
    usage: Dict[str, str]
    handler: Handler
    required_capabilities: Sequence[str] = field(default_factory=list)
    long_help: Optional[LocalizedString] = None
    rate_limit: Optional[RateLimit] = None
    allow_pipeline: bool = True

    def usage_for(self, locale: str) -> str:
        return self.usage.get(locale, self.usage.get("default", ""))

    def long_help_for(self, locale: str) -> str:
        if self.long_help:
            return self.long_help.resolve(locale)
        return self.summary.resolve(locale)


class CommandRegistry:
    def __init__(self) -> None:
        self._commands: Dict[str, Command] = {}

    def register(self, command: Command) -> None:
        self._commands[command.name] = command

    def get(self, name: str) -> Optional[Command]:
        return self._commands.get(name)

    def names(self) -> List[str]:
        return sorted(self._commands.keys())

    def values(self) -> Iterable[Command]:
        return self._commands.values()


class TranscriptLogger:
    def __init__(self, root: Path) -> None:
        self._root = root
        self._root.mkdir(parents=True, exist_ok=True)
        timestamp = now_utc().strftime("%Y%m%dT%H%M%SZ")
        self._path = self._root / f"session-{timestamp}.jsonl"
        self._file = self._path.open("a", encoding="utf-8")

    @property
    def path(self) -> Path:
        return self._path

    def log(self, payload: Dict[str, Any]) -> None:
        self._file.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()


class ShellSession:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.cwd = self.root
        self.locale = "en"
        self.registry = CommandRegistry()
        self.capabilities = {"basic", "filesystem", "process", "admin"}
        script_budget_default = 200
        env_budget = os.environ.get("OS2_SCRIPT_TOKEN_BUDGET")
        if env_budget:
            try:
                parsed_budget = int(env_budget)
                if parsed_budget > 0:
                    script_budget_default = parsed_budget
            except ValueError:
                pass
        self.config: Dict[str, Any] = {"script_token_budget": script_budget_default}
        self.user = os.environ.get("USER", "os2")
        self.start_time = now_utc()
        self.processes = ProcessTable(owner=self.user)
        self.jobs: Dict[int, Job] = {}
        self._next_job_id = 1
        self.rate_limiter = RateLimiter()
        self._lock = threading.RLock()
        self.stream_python_output = False
        logs_root = root / "rust" / "os2-kernel" / "logs" / "cli_sessions"
        self.transcript = TranscriptLogger(logs_root)
        logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
        self.logger = logging.getLogger("os2.shell")
        self.kernel_log = KernelLogWriter.for_workspace(self.root)
        registry_path = self.root / "cli" / "data" / "models.json"
        self.model_registry = ModelRegistry(registry_path)
        ledger_path = self.root / "cli" / "data" / "download_ledger.jsonl"
        self.download_ledger = TokenBudgetLedger(ledger_path)
        snapshot_ledger_path = self.root / "cli" / "data" / "snapshot_ledger.jsonl"
        self.snapshot_ledger = SnapshotLedger(snapshot_ledger_path)
        self.module_permissions = ModulePermissionRegistry(self.root, self.snapshot_ledger)
        cas_root = self.root / "cli" / "cas"
        self.cas_store = ContentAddressableStore(cas_root)
        signing_key = os.environ.get("OS2_MODEL_SIGNING_KEY", "os2-model-signing")
        self.signature_verifier = SignatureVerifier({"default": signing_key}, default_key="default")
        self.download_manager = SecureDownloadManager(
            self.download_ledger,
            snapshot_ledger=self.snapshot_ledger,
            cas=self.cas_store,
            signature_verifier=self.signature_verifier,
        )
        kernel_signing_key = os.environ.get("OS2_KERNEL_SIGNING_KEY", "os2-kernel-signing")
        kernel_signature = SignatureVerifier({"default": kernel_signing_key}, default_key="default")
        self.boot_loader = BootModelLoader(self.model_registry)
        self.boot_report = self.boot_loader.load(self.capabilities)
        self.llm_adapter = DeterministicLLMAdapter()
        register_default_feedback_hooks(self.llm_adapter, self.root)
        self.gpu_manager = GPUAccessManager(self.root, self.snapshot_ledger)
        self.runtime_loader = RuntimeLoader()
        self.ai_replay = AIReplayManager(
            self.root,
            self.snapshot_ledger,
            self.model_registry,
            self.llm_adapter,
        )
        self.ai_executor = DeterministicAIExecutor(
            self.root,
            self.model_registry,
            self.llm_adapter,
            self.kernel_log,
            self.snapshot_ledger,
            self.runtime_loader,
            replay_manager=self.ai_replay,
            gpu_manager=self.gpu_manager,
        )
        self.python_envs = PythonEnvironmentManager(self.root, self.snapshot_ledger)
        self.python_vm = PythonVMLauncher(
            self.root,
            self.snapshot_ledger,
            module_permissions=self.module_permissions,
        )
        self.self_task_review = SelfTaskReviewModule(self.root, self.snapshot_ledger)
        self.self_feedback = SelfFeedbackAnalyzer(self.root, self.snapshot_ledger)
        self.task_proposals = TaskProposalRegistry(self.root, self.snapshot_ledger)
        self.kernel_updates = KernelUpdateDistributor(
            self.root,
            self.snapshot_ledger,
            kernel_signature,
        )
        self.snapshot_benchmarks = SnapshotBenchmarkManager(
            self.root,
            self.snapshot_ledger,
        )
        self.kernel_performance = KernelPerformanceMonitor(
            self.root,
            self.snapshot_ledger,
        )
        self.deterministic_recompile = DeterministicRecompileManager(
            self.root,
            self.snapshot_ledger,
        )
        self.module_cleaner = ModuleCleaner(
            self.root,
            self.snapshot_ledger,
        )
        self.living_system = LivingDeterministicSystemManager(
            self.root,
            self.snapshot_ledger,
        )
        self.hf_client = SignedHuggingFaceClient(
            token=os.environ.get("HF_API_TOKEN"),
            signing_key=os.environ.get("OS2_HF_SIGNING_KEY", "os2-hf-signing"),
        )
        self.session_auth = SnapshotAuthenticator(self.root, self.snapshot_ledger)
        self.authenticated_session: Optional[Dict[str, Any]] = None
        self.entropy_auditor = EntropyAuditor(self.root, self.snapshot_ledger)
        self.integrity_monitor = IntegrityMonitor(self.root, self.snapshot_ledger)
        self.backup_manager = SecureBackupManager(
            self.root,
            self.snapshot_ledger,
            self.signature_verifier,
        )
        self.security_log = SecurityLogManager(self.root, self.snapshot_ledger)
        self.python_verifier = PythonDeterminismVerifier(self.root, self.snapshot_ledger)
        self.deterministic_benchmark = DeterministicBenchmarkRunner(
            self.root,
            self.snapshot_ledger,
            signature_verifier=self.signature_verifier,
            python_verifier=self.python_verifier,
            kernel_performance=self.kernel_performance,
            ai_replay=self.ai_replay,
        )
        self.documentation = DocumentationPublisher(self.root, self.snapshot_ledger)

    # -------------------- registry helpers --------------------
    def register(self, command: Command) -> None:
        self.registry.register(command)

    # -------------------- capability helpers -----------------
    def require_capabilities(self, command: Command) -> Optional[str]:
        missing = [cap for cap in command.required_capabilities if cap not in self.capabilities]
        if missing:
            return f"Missing capabilities: {', '.join(missing)}"
        return None

    # -------------------- job helpers -------------------------
    def _next_job(self) -> int:
        jid = self._next_job_id
        self._next_job_id += 1
        return jid

    def add_job(self, command_line: str, thread: threading.Thread) -> Job:
        job = Job(job_id=self._next_job(), command_line=command_line, thread=thread)
        self.jobs[job.job_id] = job
        return job

    # -------------------- execution ---------------------------
    def execute(self, invocation: CommandInvocation) -> CommandResult:
        command = self.registry.get(invocation.name)
        if not command:
            return CommandResult(status=127, stderr=f"Command not found: {invocation.name}\n")

        missing = self.require_capabilities(command)
        if missing:
            return CommandResult(status=126, stderr=missing + "\n")

        if "admin" in command.required_capabilities and self.authenticated_session is None:
            return CommandResult(
                status=1,
                stderr="Admin commands require snapshot authentication\n",
            )

        if command.rate_limit and not self.rate_limiter.check(command.name, command.rate_limit):
            return CommandResult(status=1, stderr="Rate limit exceeded. Try again later.\n")

        result = command.handler(self, invocation)
        result.audit.setdefault("command", invocation.name)
        result.audit.setdefault("args", invocation.args)
        result.audit.setdefault("status", result.status)
        result.audit.setdefault("timestamp", isoformat_utc(now_utc()))
        result.audit.setdefault("capabilities", sorted(self.capabilities))
        return result

    # -------------------- line execution ----------------------
    def run_line(self, line: str, background: bool = False) -> CommandResult:
        with self._lock:
            pipeline = [segment.strip() for segment in line.split("|")]
            stdin: Optional[str] = None
            result = CommandResult()
            for index, segment in enumerate(pipeline):
                args = shlex.split(segment)
                if not args:
                    continue
                invocation = CommandInvocation(
                    name=args[0],
                    args=args[1:],
                    stdin=stdin,
                    pipeline_index=index,
                    pipeline_length=len(pipeline),
                )
                piece = self.execute(invocation)
                self._audit(piece)
                result.merge(piece)
                stdin = piece.stdout
                if piece.status != 0:
                    break
            return result

    def _audit(self, result: CommandResult) -> None:
        payload = {
            "ts": isoformat_utc(now_utc()),
            "user": self.user,
            "cwd": str(self.cwd),
            **result.audit,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
        self.transcript.log(payload)
        try:
            self.self_feedback.ingest(payload)
        except SelfFeedbackError as exc:
            self.logger.error("Failed to record self-feedback: %s", exc)

    # -------------------- utilities ---------------------------
    def change_directory(self, target: Path) -> CommandResult:
        resolved = (self.cwd / target).resolve()
        if not str(resolved).startswith(str(self.root)):
            return CommandResult(status=1, stderr="Permission denied: outside sandbox\n")
        if not resolved.exists() or not resolved.is_dir():
            return CommandResult(status=1, stderr=f"No such directory: {resolved}\n")
        self.cwd = resolved
        return CommandResult(stdout=str(self.cwd) + "\n")

    def uptime(self) -> _dt.timedelta:
        return now_utc() - self.start_time

    def close(self) -> None:
        try:
            self._merge_python_state()
        except SnapshotLedgerError as exc:
            self.logger.error("Failed to merge Python state with snapshot ledger: %s", exc)
        self.transcript.close()

    def _merge_python_state(self) -> None:
        sandboxes_dir = self.root / "cli" / "python_vm" / "sandboxes"
        sessions_dir = self.root / "cli" / "python_vm" / "snapshots" / "sessions"
        digest = hashlib.sha256()
        files_hashed = 0

        for directory in (sandboxes_dir, sessions_dir):
            if not directory.exists():
                continue
            for path in sorted(directory.rglob("*.json")):
                digest.update(path.read_bytes())
                files_hashed += 1

        if files_hashed == 0:
            return

        self.snapshot_ledger.record_event(
            {
                "kind": "python_state_merged",
                "files_hashed": files_hashed,
                "state_digest": digest.hexdigest(),
            }
        )


# ---------------------------------------------------------------------------
# Built-in command implementations
# ---------------------------------------------------------------------------


def command(name: str, summary: str, usage: str, **kwargs: Any) -> Callable[[Handler], Handler]:
    loc = kwargs.pop("localizations", {})
    long_help = kwargs.pop("long_help", None)
    capability = kwargs.pop("capabilities", ("basic",))
    rate_limit = kwargs.pop("rate_limit", None)
    allow_pipeline = kwargs.pop("allow_pipeline", True)

    def decorator(func: Handler) -> Handler:
        func.__command_definition__ = Command(
            name=name,
            summary=LocalizedString(summary, loc),
            usage={"default": usage, **{k: v for k, v in loc.items()}},
            handler=func,
            required_capabilities=capability,
            long_help=LocalizedString(long_help or summary, loc) if long_help else None,
            rate_limit=rate_limit,
            allow_pipeline=allow_pipeline,
        )
        return func

    return decorator


# -------------------- system commands -----------------------


@command(
    name="help",
    summary="List available commands",
    usage="help [command]",
)
def help_command(shell: ShellSession, invocation: CommandInvocation) -> CommandResult:
    if not invocation.args:
        entries = [
            f"{name:15s} - {shell.registry.get(name).summary.resolve(shell.locale)}"
            for name in shell.registry.names()
        ]
        return CommandResult(stdout="\n".join(entries) + "\n")
    name = invocation.args[0]
    command = shell.registry.get(name)
    if not command:
        return CommandResult(status=1, stderr=f"Unknown command: {name}\n")
    body = f"{name}\n{'-' * len(name)}\n{command.long_help_for(shell.locale)}\nUsage: {command.usage_for(shell.locale)}\n"
    return CommandResult(stdout=body)


@command(
    name="man",
    summary="Display detailed command documentation",
    usage="man <command>",
)
def man_command(shell: ShellSession, invocation: CommandInvocation) -> CommandResult:
    if not invocation.args:
        return CommandResult(status=1, stderr="Usage: man <command>\n")
    name = invocation.args[0]
    command = shell.registry.get(name)
    if not command:
        return CommandResult(status=1, stderr=f"No manual entry for {name}\n")
    lines = [
        f"NAME\n    {name} - {command.summary.resolve(shell.locale)}",
        f"SYNOPSIS\n    {command.usage_for(shell.locale)}",
        f"DESCRIPTION\n    {command.long_help_for(shell.locale)}",
        f"CAPABILITIES\n    {', '.join(command.required_capabilities) or 'None'}",
    ]
    return CommandResult(stdout="\n\n".join(lines) + "\n")


@command(
    name="set-locale",
    summary="Switch shell locale",
    usage="set-locale <locale>",
)
def set_locale(shell: ShellSession, invocation: CommandInvocation) -> CommandResult:
    if not invocation.args:
        return CommandResult(status=1, stderr="Usage: set-locale <locale>\n")
    locale = invocation.args[0]
    shell.locale = locale
    return CommandResult(stdout=f"Locale set to {locale}\n")


# -------------------- filesystem commands -------------------


def _format_listing(path: Path) -> str:
    entries = []
    for item in sorted(path.iterdir()):
        suffix = "/" if item.is_dir() else ""
        entries.append(item.name + suffix)
    return "  ".join(entries)


@command(
    name="pwd",
    summary="Print working directory",
    usage="pwd",
    capabilities=("filesystem",),
)
def pwd(shell: ShellSession, _: CommandInvocation) -> CommandResult:
    return CommandResult(stdout=str(shell.cwd) + "\n")


@command(
    name="ls",
    summary="List directory contents",
    usage="ls [path]",
    capabilities=("filesystem",),
)
def ls(shell: ShellSession, invocation: CommandInvocation) -> CommandResult:
    target = shell.cwd if not invocation.args else (shell.cwd / invocation.args[0])
    target = target.resolve()
    if not str(target).startswith(str(shell.root)):
        return CommandResult(status=1, stderr="Permission denied\n")
    if not target.exists():
        return CommandResult(status=1, stderr="No such file or directory\n")
    if target.is_file():
        return CommandResult(stdout=target.name + "\n")
    return CommandResult(stdout=_format_listing(target) + "\n")


@command(
    name="cd",
    summary="Change directory",
    usage="cd <path>",
    capabilities=("filesystem",),
)
def cd(shell: ShellSession, invocation: CommandInvocation) -> CommandResult:
    if not invocation.args:
        return CommandResult(status=1, stderr="Usage: cd <path>\n")
    return shell.change_directory(Path(invocation.args[0]))


@command(
    name="mkdir",
    summary="Create directory",
    usage="mkdir <path>",
    capabilities=("filesystem",),
)
def mkdir(shell: ShellSession, invocation: CommandInvocation) -> CommandResult:
    if not invocation.args:
        return CommandResult(status=1, stderr="Usage: mkdir <path>\n")
    target = (shell.cwd / invocation.args[0]).resolve()
    if not str(target).startswith(str(shell.root)):
        return CommandResult(status=1, stderr="Permission denied\n")
    target.mkdir(parents=True, exist_ok=True)
    return CommandResult(stdout=f"Created {target}\n")


@command(
    name="touch",
    summary="Create empty file",
    usage="touch <path>",
    capabilities=("filesystem",),
)
def touch(shell: ShellSession, invocation: CommandInvocation) -> CommandResult:
    if not invocation.args:
        return CommandResult(status=1, stderr="Usage: touch <path>\n")
    target = (shell.cwd / invocation.args[0]).resolve()
    if not str(target).startswith(str(shell.root)):
        return CommandResult(status=1, stderr="Permission denied\n")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.touch(exist_ok=True)
    return CommandResult(stdout=f"Touched {target}\n")


@command(
    name="rm",
    summary="Remove file",
    usage="rm <path>",
    capabilities=("filesystem",),
    rate_limit=RateLimit(limit=5, per_seconds=60),
)
def rm(shell: ShellSession, invocation: CommandInvocation) -> CommandResult:
    if not invocation.args:
        return CommandResult(status=1, stderr="Usage: rm <path>\n")
    target = (shell.cwd / invocation.args[0]).resolve()
    if not str(target).startswith(str(shell.root)):
        return CommandResult(status=1, stderr="Permission denied\n")
    if target.is_dir():
        return CommandResult(status=1, stderr="rm only supports files\n")
    if not target.exists():
        return CommandResult(status=1, stderr="No such file\n")
    target.unlink()
    return CommandResult(stdout=f"Removed {target}\n")


@command(
    name="mv",
    summary="Move or rename path",
    usage="mv <source> <destination>",
    capabilities=("filesystem",),
)
def mv(shell: ShellSession, invocation: CommandInvocation) -> CommandResult:
    if len(invocation.args) != 2:
        return CommandResult(status=1, stderr="Usage: mv <src> <dst>\n")
    src = (shell.cwd / invocation.args[0]).resolve()
    dst = (shell.cwd / invocation.args[1]).resolve()
    if not str(src).startswith(str(shell.root)) or not str(dst).startswith(str(shell.root)):
        return CommandResult(status=1, stderr="Permission denied\n")
    if not src.exists():
        return CommandResult(status=1, stderr="Source missing\n")
    dst.parent.mkdir(parents=True, exist_ok=True)
    src.replace(dst)
    return CommandResult(stdout=f"Moved {src} -> {dst}\n")


@command(
    name="cp",
    summary="Copy file",
    usage="cp <source> <destination>",
    capabilities=("filesystem",),
)
def cp(shell: ShellSession, invocation: CommandInvocation) -> CommandResult:
    if len(invocation.args) != 2:
        return CommandResult(status=1, stderr="Usage: cp <src> <dst>\n")
    src = (shell.cwd / invocation.args[0]).resolve()
    dst = (shell.cwd / invocation.args[1]).resolve()
    if not str(src).startswith(str(shell.root)) or not str(dst).startswith(str(shell.root)):
        return CommandResult(status=1, stderr="Permission denied\n")
    if not src.exists() or not src.is_file():
        return CommandResult(status=1, stderr="Source must be a file\n")
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(src.read_bytes())
    return CommandResult(stdout=f"Copied {src} -> {dst}\n")


# -------------------- process management --------------------


def _format_process(proc: Process) -> str:
    uptime = (now_utc() - proc.started_at).total_seconds()
    return f"{proc.pid:5d} {proc.user:8s} {proc.state:10s} {proc.cpu_percent:6.2f} {proc.memory_kb:7d} {uptime:7.1f} {proc.name}"


@command(
    name="ps",
    summary="List running processes",
    usage="ps",
    capabilities=("process",),
)
def ps(shell: ShellSession, _: CommandInvocation) -> CommandResult:
    header = " PID   USER     STATE         CPU    MEMORY   UPTIME NAME"
    lines = [header] + [_format_process(proc) for proc in shell.processes.all()]
    return CommandResult(stdout="\n".join(lines) + "\n")


@command(
    name="top",
    summary="Show top resource consumers",
    usage="top [count]",
    capabilities=("process",),
)
def top(shell: ShellSession, invocation: CommandInvocation) -> CommandResult:
    count = 5
    if invocation.args:
        try:
            count = int(invocation.args[0])
        except ValueError:
            return CommandResult(status=1, stderr="Count must be numeric\n")
    processes = sorted(shell.processes.all(), key=lambda p: p.cpu_percent, reverse=True)[:count]
    header = " PID   USER     STATE         CPU    MEMORY   NAME"
    lines = [header]
    for proc in processes:
        lines.append(
            f"{proc.pid:5d} {proc.user:8s} {proc.state:10s} {proc.cpu_percent:6.2f} {proc.memory_kb:7d} {proc.name}"
        )
    return CommandResult(stdout="\n".join(lines) + "\n")


@command(
    name="kill",
    summary="Terminate a process",
    usage="kill <pid>",
    capabilities=("process", "admin"),
    rate_limit=RateLimit(limit=5, per_seconds=60),
)
def kill(shell: ShellSession, invocation: CommandInvocation) -> CommandResult:
    if not invocation.args:
        return CommandResult(status=1, stderr="Usage: kill <pid>\n")
    try:
        pid = int(invocation.args[0])
    except ValueError:
        return CommandResult(status=1, stderr="PID must be numeric\n")
    if shell.processes.kill(pid):
        return CommandResult(stdout=f"Killed {pid}\n")
    return CommandResult(status=1, stderr="No such process\n")


@command(
    name="jobs",
    summary="List background jobs",
    usage="jobs",
)
def jobs(shell: ShellSession, _: CommandInvocation) -> CommandResult:
    if not shell.jobs:
        return CommandResult(stdout="No background jobs\n")
    lines = []
    for job in sorted(shell.jobs.values(), key=lambda j: j.job_id):
        lines.append(f"[{job.job_id}] {job.status:10s} {job.command_line}")
    return CommandResult(stdout="\n".join(lines) + "\n")


@command(
    name="fg",
    summary="Bring job to foreground",
    usage="fg <job_id>",
)
def fg(shell: ShellSession, invocation: CommandInvocation) -> CommandResult:
    if not invocation.args:
        return CommandResult(status=1, stderr="Usage: fg <job_id>\n")
    try:
        job_id = int(invocation.args[0])
    except ValueError:
        return CommandResult(status=1, stderr="Job id must be numeric\n")
    job = shell.jobs.get(job_id)
    if not job:
        return CommandResult(status=1, stderr="No such job\n")
    if job.thread and job.thread.is_alive():
        job.thread.join()
    job.status = "completed"
    output = job.result.stdout if job.result else ""
    stderr = job.result.stderr if job.result else ""
    return CommandResult(stdout=output, stderr=stderr, status=job.result.status if job.result else 0)


@command(
    name="bg",
    summary="Resume a stopped job in background",
    usage="bg <job_id>",
)
def bg(shell: ShellSession, invocation: CommandInvocation) -> CommandResult:
    if not invocation.args:
        return CommandResult(status=1, stderr="Usage: bg <job_id>\n")
    try:
        job_id = int(invocation.args[0])
    except ValueError:
        return CommandResult(status=1, stderr="Job id must be numeric\n")
    job = shell.jobs.get(job_id)
    if not job:
        return CommandResult(status=1, stderr="No such job\n")
    if job.status == "completed":
        return CommandResult(stdout=f"Job {job_id} already completed\n")
    return CommandResult(stdout=f"Job {job_id} running in background\n")


# -------------------- system introspection ------------------


@command(
    name="uname",
    summary="Display system information",
    usage="uname",
)
def uname(shell: ShellSession, _: CommandInvocation) -> CommandResult:
    return CommandResult(stdout=f"{OS_NAME} Kernel (deterministic)\n")


@command(
    name="whoami",
    summary="Show current user",
    usage="whoami",
)
def whoami(shell: ShellSession, _: CommandInvocation) -> CommandResult:
    return CommandResult(stdout=shell.user + "\n")


@command(
    name="env",
    summary="List environment variables",
    usage="env",
)
def env(shell: ShellSession, _: CommandInvocation) -> CommandResult:
    lines = [f"{key}={value}" for key, value in sorted(os.environ.items())]
    return CommandResult(stdout="\n".join(lines) + "\n")


@command(
    name="date",
    summary="Show current time",
    usage="date",
)
def date(shell: ShellSession, _: CommandInvocation) -> CommandResult:
    return CommandResult(stdout=isoformat_utc(now_utc()) + "\n")


@command(
    name="uptime",
    summary="Show shell uptime",
    usage="uptime",
)
def uptime(shell: ShellSession, _: CommandInvocation) -> CommandResult:
    delta = shell.uptime()
    return CommandResult(stdout=str(delta) + "\n")


# -------------------- capabilities and configuration --------


@command(
    name="cap-list",
    summary="List granted capabilities",
    usage="cap-list",
)
def cap_list(shell: ShellSession, _: CommandInvocation) -> CommandResult:
    caps = ", ".join(sorted(shell.capabilities)) or "(none)"
    return CommandResult(stdout=caps + "\n")


@command(
    name="cap-grant",
    summary="Grant a capability",
    usage="cap-grant <capability>",
    capabilities=("admin",),
    rate_limit=RateLimit(limit=5, per_seconds=60),
)
def cap_grant(shell: ShellSession, invocation: CommandInvocation) -> CommandResult:
    if not invocation.args:
        return CommandResult(status=1, stderr="Usage: cap-grant <capability>\n")
    capability = invocation.args[0]
    shell.capabilities.add(capability)
    return CommandResult(stdout=f"Granted {capability}\n")


@command(
    name="config-get",
    summary="Read configuration value",
    usage="config-get <key>",
)
def config_get(shell: ShellSession, invocation: CommandInvocation) -> CommandResult:
    if not invocation.args:
        return CommandResult(status=1, stderr="Usage: config-get <key>\n")
    key = invocation.args[0]
    value = shell.config.get(key)
    if value is None:
        return CommandResult(status=1, stderr="No such config key\n")
    return CommandResult(stdout=f"{key}={value}\n")


@command(
    name="config-set",
    summary="Update configuration value",
    usage="config-set <key> <value>",
    capabilities=("admin",),
    rate_limit=RateLimit(limit=5, per_seconds=60),
)
def config_set(shell: ShellSession, invocation: CommandInvocation) -> CommandResult:
    if len(invocation.args) != 2:
        return CommandResult(status=1, stderr="Usage: config-set <key> <value>\n")
    key, value = invocation.args
    shell.config[key] = value
    return CommandResult(stdout=f"Updated {key}\n")


@command(
    name="snapshot-auth",
    summary="Authenticate admin commands with a snapshot identity",
    usage="snapshot-auth <snapshot_id> [--reason TEXT] [--json]",
)
def snapshot_auth(shell: ShellSession, invocation: CommandInvocation) -> CommandResult:
    parser = argparse.ArgumentParser(prog="snapshot-auth", add_help=False)
    parser.add_argument("snapshot_id", type=int)
    parser.add_argument("--reason")
    parser.add_argument("--json", action="store_true", dest="as_json")
    try:
        parsed = parser.parse_args(invocation.args)
    except SystemExit:
        return _usage_error("Usage: snapshot-auth <snapshot_id> [--reason TEXT] [--json]")

    try:
        record = shell.session_auth.authenticate(
            user=shell.user,
            snapshot_id=parsed.snapshot_id,
            reason=parsed.reason,
        )
    except SessionAuthenticationError as exc:
        return CommandResult(status=1, stderr=str(exc) + "\n")

    shell.authenticated_session = record
    shell.capabilities.add("snapshot-authenticated")
    audit = {
        "snapshot_id": record.get("snapshot_id"),
        "session_token": record.get("session_token"),
        "ledger_event_id": record.get("ledger_event_id"),
    }
    if parsed.as_json:
        body = json.dumps(record, indent=2, ensure_ascii=False) + "\n"
        return CommandResult(stdout=body, audit=audit)

    message = (
        "Authenticated snapshot "
        + str(record.get("snapshot_id"))
        + " token="
        + str(record.get("session_token"))
    )
    ledger_event = record.get("ledger_event_id")
    if ledger_event:
        message += f" (ledger={ledger_event})"
    return CommandResult(stdout=message + "\n", audit=audit)


# -------------------- scripting and plugins -----------------


def execute_pipeline(shell: ShellSession, line: str) -> CommandResult:
    return shell.run_line(line)


def _script_lines(path: Path) -> List[str]:
    return [line.rstrip() for line in path.read_text(encoding="utf-8").splitlines()]


def _should_execute(stack: List[Dict[str, Any]]) -> bool:
    return not stack or stack[-1]["execute"]


@command(
    name="run-script",
    summary="Execute batch script",
    usage="run-script <path>",
)
def run_script(shell: ShellSession, invocation: CommandInvocation) -> CommandResult:
    if not invocation.args:
        return CommandResult(status=1, stderr="Usage: run-script <path>\n")
    script_path = (shell.cwd / invocation.args[0]).resolve()
    if not str(script_path).startswith(str(shell.root)):
        return CommandResult(status=1, stderr="Permission denied\n")
    if not script_path.exists():
        return CommandResult(status=1, stderr="Script not found\n")
    lines = _script_lines(script_path)
    budget = int(shell.config.get("script_token_budget", 200))
    executed = 0
    stack: List[Dict[str, Any]] = []
    for raw in lines:
        if executed >= budget:
            return CommandResult(status=1, stderr="Script token budget exceeded\n")
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.lower().startswith("if "):
            condition_line = line[3:].strip()
            result = execute_pipeline(shell, condition_line)
            execute_branch = result.status == 0
            stack.append({"type": "if", "execute": execute_branch, "condition": execute_branch, "else_consumed": False})
            executed += 1
            continue
        if line.lower() == "else":
            if not stack or stack[-1]["type"] != "if":
                return CommandResult(status=1, stderr="Unexpected else\n")
            frame = stack[-1]
            frame["execute"] = not frame["condition"]
            frame["else_consumed"] = True
            continue
        if line.lower() == "endif":
            if not stack or stack[-1]["type"] != "if":
                return CommandResult(status=1, stderr="Unexpected endif\n")
            stack.pop()
            continue
        if not _should_execute(stack):
            continue
        executed += 1
        result = execute_pipeline(shell, line)
        if result.status != 0:
            return result
    if stack:
        return CommandResult(status=1, stderr="Unclosed conditional block\n")
    return CommandResult(stdout=f"Script {script_path} executed successfully\n")


class _PythonCommandParseError(ValueError):
    """Internal error used to signal invalid Python command usage."""


def _parse_python_command_args(
    args: Sequence[str], *, usage: str, allow_dash_c: bool
) -> Tuple[
    Optional[str],
    Optional[str],
    Optional[str],
    List[str],
    bool,
    Optional[int],
    Optional[int],
    bool,
]:
    expr: Optional[str] = None
    script: Optional[str] = None
    module: Optional[str] = None
    script_args: List[str] = []
    json_output = False
    resume_snapshot: Optional[int] = None
    token_budget_override: Optional[int] = None
    safe_mode = False
    idx = 0

    while idx < len(args):
        token = args[idx]
        if token == "--json":
            json_output = True
        elif token == "--safe":
            safe_mode = True
        elif token == "--resume":
            idx += 1
            if idx >= len(args):
                raise _PythonCommandParseError("--resume requires a snapshot identifier")
            value = args[idx]
            if resume_snapshot is not None:
                raise _PythonCommandParseError("Only one --resume value may be provided")
            try:
                resume_snapshot = int(value)
            except ValueError as exc:
                raise _PythonCommandParseError("--resume expects an integer snapshot identifier") from exc
            if resume_snapshot <= 0:
                raise _PythonCommandParseError("--resume expects a positive snapshot identifier")
        elif token in ("--version", "-V"):
            if expr is not None or script is not None or module is not None:
                raise _PythonCommandParseError("--version cannot be combined with other sources")
            expr = "\"Python \" + __import__('sys').version"
        elif token in ("--module",) or (allow_dash_c and token == "-m"):
            idx += 1
            if idx >= len(args):
                raise _PythonCommandParseError("-m/--module requires a module name")
            if expr is not None or script is not None or module is not None:
                raise _PythonCommandParseError("Cannot combine module execution with other sources")
            module = args[idx]
        elif token == "--eval" or (allow_dash_c and token == "-c"):
            idx += 1
            if idx >= len(args):
                flag = "-c" if token == "-c" else "--eval"
                raise _PythonCommandParseError(f"{flag} requires code to execute")
            if script is not None:
                raise _PythonCommandParseError("Cannot combine code flags with a script path")
            if module is not None:
                raise _PythonCommandParseError("Cannot combine code flags with a module name")
            expr = args[idx]
        elif token in ("--token-budget", "--budget"):
            idx += 1
            if idx >= len(args):
                raise _PythonCommandParseError("--token-budget requires an integer value")
            value = args[idx]
            if token_budget_override is not None:
                raise _PythonCommandParseError("Only one --token-budget value may be provided")
            try:
                token_budget_override = int(value)
            except ValueError as exc:
                raise _PythonCommandParseError("--token-budget expects a positive integer") from exc
            if token_budget_override <= 0:
                raise _PythonCommandParseError("--token-budget expects a positive integer")
        elif token == "--":
            script_args = list(args[idx + 1 :])
            break
        elif script is None and expr is None and module is None:
            script = token
        elif script is not None:
            script_args = list(args[idx:])
            break
        elif expr is not None:
            script_args = list(args[idx:])
            break
        elif module is not None:
            script_args = list(args[idx:])
            break
        else:  # pragma: no cover - defensive (should not reach)
            raise _PythonCommandParseError(f"Unexpected argument: {token}")
        idx += 1

    if expr is None and script is None and module is None and resume_snapshot is None:
        raise _PythonCommandParseError(f"Usage: {usage}")

    return (
        expr,
        script,
        module,
        script_args,
        json_output,
        resume_snapshot,
        token_budget_override,
        safe_mode,
    )


def _run_python_vm_command(
    shell: ShellSession,
    invocation: CommandInvocation,
    *,
    usage: str,
    allow_dash_c: bool,
) -> CommandResult:
    try:
        (
            expr,
            script,
            module,
            script_args,
            json_output,
            resume_snapshot,
            token_budget_override,
            safe_mode,
        ) = _parse_python_command_args(invocation.args, usage=usage, allow_dash_c=allow_dash_c)
    except _PythonCommandParseError as exc:
        return _usage_error(str(exc))

    script_path: Optional[Path] = None
    if script is not None:
        candidate = (shell.cwd / script).resolve()
        if not str(candidate).startswith(str(shell.root)):
            return CommandResult(status=1, stderr="Permission denied\n")
        script_path = candidate

    stream_output = (
        bool(getattr(shell, "stream_python_output", False))
        and not json_output
        and invocation.pipeline_length == 1
        and invocation.stdin is None
    )

    default_budget = shell.config.get("script_token_budget")
    try:
        default_budget_int = int(default_budget) if default_budget is not None else 200
    except (TypeError, ValueError):
        default_budget_int = 200
    token_budget = token_budget_override or default_budget_int

    launch_kwargs = {
        "expr": expr,
        "script_path": script_path,
        "module": module,
        "script_args": script_args,
        "token_budget": token_budget,
        "capabilities": sorted(shell.capabilities),
        "command_alias": invocation.name,
        "resume_snapshot": resume_snapshot,
        "safe_mode": safe_mode,
    }
    if stream_output:
        launch_kwargs["stream_output"] = True

    try:
        result = shell.python_vm.launch(**launch_kwargs)
    except TypeError as exc:
        # ``PythonVMLauncher.launch`` gained the ``stream_output`` keyword when
        # interactive teeing support shipped. Older shells may still import a
        # launcher that has not yet been upgraded, raising a ``TypeError`` when
        # the new keyword is provided. Fall back to calling the legacy
        # signature so those environments continue to function, albeit without
        # interactive streaming.
        if "stream_output" in str(exc) and "stream_output" in launch_kwargs:
            launch_kwargs.pop("stream_output", None)
            result = shell.python_vm.launch(**launch_kwargs)
            stream_output = False
        else:  # pragma: no cover - defensive
            raise
    except PythonVMError as exc:
        return CommandResult(status=1, stderr=f"Python VM error: {exc}\n")

    syspath_event = result.events.get("syspath") or {}
    async_queue_summary = result.events.get("async_queue") or {}
    sys_path = [entry.get("raw") for entry in syspath_event.get("paths", [])]
    events = {
        "start": result.events["start"].get("event_id"),
        "complete": result.events["complete"].get("event_id"),
        "sandbox_created": result.events["sandbox_created"].get("event_id")
        if result.events.get("sandbox_created")
        else None,
        "sandbox_released": result.events["sandbox_released"].get("event_id"),
        "syspath": syspath_event.get("event_id"),
        "snapshot_tag": result.events.get("snapshot_tag", {}).get("event_id"),
        "async_queue_created": async_queue_summary.get("ledger_event_ids", {}).get("created"),
        "async_queue_drained": async_queue_summary.get("ledger_event_ids", {}).get("drained"),
        "resume_request": (
            result.events.get("resume_request", {}).get("event_id")
            if isinstance(result.events.get("resume_request"), Mapping)
            else None
        ),
        "resume_hydrated": (
            result.events.get("resume_hydrated", {}).get("event_id")
            if isinstance(result.events.get("resume_hydrated"), Mapping)
            else None
        ),
        "snapshot_state_saved": (
            result.events.get("snapshot_state", {}).get("event_id")
            if isinstance(result.events.get("snapshot_state"), Mapping)
            else None
        ),
    }

    kernel_log_events = result.events.get("kernel_log") or {}
    kernel_log_refs = {
        name: {
            "chain_hash": event.get("chain_hash"),
            "timestamp": event.get("timestamp"),
            "sequence": event.get("sequence"),
        }
        for name, event in kernel_log_events.items()
        if event
    }

    shell.transcript.log(
        {
            "type": "python_vm",
            "command": invocation.name,
            "session_id": result.session_id,
            "sandbox_id": result.sandbox_id,
            "snapshot_id": result.snapshot_id,
            "mode": result.mode,
            "module": result.module,
            "status": "ok" if result.exit_status == 0 else "error",
            "duration_ms": round(result.duration_ms, 3),
            "stdout": result.stdout,
            "stderr": result.stderr,
            "script": str(script_path) if script_path else None,
            "args": list(script_args),
            "token_budget": result.token_budget,
            "tokens_consumed": result.tokens_consumed,
            "events": events,
            "sys_path": sys_path,
            "sys_path_hash": syspath_event.get("paths_hash"),
            "kernel_log": kernel_log_refs,
            "kernel_log_path": str(shell.python_vm.kernel_log_path),
            "snapshot_tag_event_id": result.events.get("snapshot_tag", {}).get("event_id"),
            "async_queue": async_queue_summary,
            "resume_from_snapshot": result.resume_from_snapshot,
            "safe_mode": result.safe_mode,
            "snapshot_state_keys": list(result.snapshot_state_keys),
            "snapshot_state_path": result.snapshot_state_path,
        }
    )

    audit = {
        "session_id": result.session_id,
        "sandbox_id": result.sandbox_id,
        "snapshot_id": result.snapshot_id,
        "mode": result.mode,
        "module": result.module,
        "duration_ms": round(result.duration_ms, 3),
        "events": events,
        "script": str(script_path) if script_path else None,
        "args": list(script_args),
        "token_budget": result.token_budget,
        "tokens_consumed": result.tokens_consumed,
        "command_alias": invocation.name,
        "sys_path": sys_path,
        "sys_path_hash": syspath_event.get("paths_hash"),
        "kernel_log": kernel_log_refs,
        "kernel_log_path": str(shell.python_vm.kernel_log_path),
        "snapshot_tag_event_id": result.events.get("snapshot_tag", {}).get("event_id"),
        "async_queue": async_queue_summary,
        "resume_from_snapshot": result.resume_from_snapshot,
        "safe_mode": result.safe_mode,
        "snapshot_state_keys": list(result.snapshot_state_keys),
        "snapshot_state_path": result.snapshot_state_path,
    }

    if json_output:
        payload = {
            **audit,
            "status": "ok" if result.exit_status == 0 else "error",
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
        body = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
        return CommandResult(stdout=body, status=result.exit_status, audit=audit)

    stdout = result.stdout
    if stdout and not stdout.endswith("\n"):
        stdout += "\n"
    queue_status = str(async_queue_summary.get("status", "none"))
    queue_total = int(async_queue_summary.get("tasks_total", 0))
    queue_errors = int(async_queue_summary.get("tasks_error", 0))
    resume_fragment = (
        f" resume={result.resume_from_snapshot}" if result.resume_from_snapshot else ""
    )
    safe_fragment = " safe" if result.safe_mode else ""
    summary = (
        f"[{invocation.name}:{result.session_id}@{result.sandbox_id}] "
        f"mode={result.mode} status={'ok' if result.exit_status == 0 else 'error'} "
        f"duration={result.duration_ms:.2f}ms tokens={result.tokens_consumed}/{result.token_budget} "
        f"snapshot={result.snapshot_id}{resume_fragment}{safe_fragment} "
        f"async_queue={queue_status}({queue_errors}/{queue_total})"
    )
    stdout += summary + "\n"
    result_streamed = getattr(result, "streamed", False)
    if result_streamed:
        print(summary)
    return CommandResult(
        stdout=stdout,
        stderr=result.stderr,
        status=result.exit_status,
        audit=audit,
        streamed=result_streamed,
    )


# ---------------------------------------------------------------------------
# Git helper commands
# ---------------------------------------------------------------------------


def _normalize_process_output(value: str) -> str:
    if not value:
        return ""
    return value if value.endswith("\n") else value + "\n"


def _git_run(
    shell: ShellSession,
    args: Sequence[str],
    *,
    invocation: Optional[CommandInvocation] = None,
) -> CommandResult:
    stream_output = False
    if invocation is not None:
        stream_output = (
            bool(getattr(shell, "stream_python_output", False))
            and invocation.pipeline_length == 1
            and invocation.stdin is None
        )

    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    stdout_mirror: Optional[io.TextIOBase] = sys.stdout if stream_output else None
    stderr_mirror: Optional[io.TextIOBase] = sys.stderr if stream_output else None

    try:
        process = subprocess.Popen(
            ["git", *args],
            cwd=str(shell.cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
    except FileNotFoundError:
        return CommandResult(status=1, stderr="git executable not available\n")

    def _pump_stream(
        pipe: Optional[io.TextIOBase],
        buffer: io.StringIO,
        mirror: Optional[io.TextIOBase],
    ) -> None:
        if pipe is None:
            return
        try:
            while True:
                chunk = pipe.read(1)
                if not chunk:
                    break
                buffer.write(chunk)
                if mirror is not None:
                    mirror.write(chunk)
                    mirror.flush()
        finally:
            try:
                pipe.close()
            except Exception:  # pragma: no cover - defensive cleanup
                pass

    threads: List[threading.Thread] = []
    stdout_thread = threading.Thread(
        target=_pump_stream,
        args=(process.stdout, stdout_buffer, stdout_mirror),
        daemon=True,
    )
    stdout_thread.start()
    threads.append(stdout_thread)
    stderr_thread = threading.Thread(
        target=_pump_stream,
        args=(process.stderr, stderr_buffer, stderr_mirror),
        daemon=True,
    )
    stderr_thread.start()
    threads.append(stderr_thread)

    returncode: Optional[int] = None
    interrupted = False
    try:
        returncode = process.wait()
    except KeyboardInterrupt:
        interrupted = True
        try:
            process.send_signal(signal.SIGINT)
        except Exception:  # pragma: no cover - defensive
            pass
        try:
            returncode = process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                returncode = process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                returncode = process.wait()
    finally:
        for thread in threads:
            thread.join()

    stdout = _normalize_process_output(stdout_buffer.getvalue())
    stderr = _normalize_process_output(stderr_buffer.getvalue())
    audit = {
        "command": ["git", *args],
        "returncode": int(returncode if returncode is not None else process.returncode or 0),
        "cwd": str(shell.cwd),
    }
    if interrupted:
        audit["interrupted"] = True
    status = int(returncode if returncode is not None else process.returncode or 0)
    return CommandResult(
        stdout=stdout,
        stderr=stderr,
        status=status,
        audit=audit,
        streamed=stream_output,
    )


def _git_clone(
    shell: ShellSession,
    args: Sequence[str],
    *,
    invocation: Optional[CommandInvocation] = None,
) -> CommandResult:
    if not args:
        return _usage_error("Usage: git clone <url> [directory]")

    url = args[0]
    remaining = list(args[1:])
    destination: Optional[str] = None

    if "--" in remaining:
        idx = remaining.index("--")
        tail = remaining[idx + 1 :]
        if len(tail) > 1:
            return CommandResult(status=1, stderr="git clone accepts at most one destination path\n")
        destination = tail[0] if tail else None
        remaining = remaining[:idx]
    elif remaining:
        potential_dest = remaining[-1]
        if (
            not potential_dest.startswith("-")
            and (not remaining[:-1] or remaining[-2] not in _GIT_CLONE_VALUE_OPTIONS)
        ):
            destination = potential_dest
            remaining = remaining[:-1]

    parsed = urllib.parse.urlparse(url)
    if parsed.scheme in {"https", "http"}:
        if not parsed.netloc:
            return CommandResult(status=1, stderr="Clone URL must include a host\n")
    elif parsed.scheme in {"", "file"}:
        source_path = Path(parsed.path if parsed.scheme == "file" else url)
        if not source_path.is_absolute():
            source_path = (shell.cwd / source_path).resolve()
        else:
            source_path = source_path.resolve()
        if not source_path.exists():
            return CommandResult(status=1, stderr="Clone source not found\n")
    else:
        return CommandResult(status=1, stderr="Unsupported clone URL scheme\n")

    if destination:
        dest_path = (shell.cwd / destination).resolve()
        if not str(dest_path).startswith(str(shell.root)):
            return CommandResult(status=1, stderr="Permission denied\n")

    clone_args: List[str] = ["clone", url, *remaining]
    if destination:
        clone_args.append(destination)

    result = _git_run(shell, clone_args, invocation=invocation)
    audit = result.audit
    audit.setdefault("url", url)
    if destination:
        audit.setdefault("destination", str((shell.cwd / destination).resolve()))
    else:
        repo_name = url.rstrip("/").rsplit("/", 1)[-1]
        if repo_name.endswith(".git"):
            repo_name = repo_name[:-4]
        audit.setdefault("destination", str((shell.cwd / repo_name).resolve()))
    return result


@command(
    name="create-env",
    summary="Create a deterministic Python environment for sandboxed sessions",
    usage="create-env <name> [--description TEXT] [--json] [--no-pip|--with-pip]",
    capabilities=("process",),
)
def create_env_command(shell: ShellSession, invocation: CommandInvocation) -> CommandResult:
    parser = argparse.ArgumentParser(prog="create-env", add_help=False)
    parser.add_argument("name")
    parser.add_argument("--description", dest="description")
    parser.add_argument("--json", action="store_true", dest="as_json")
    pip_group = parser.add_mutually_exclusive_group()
    pip_group.add_argument("--with-pip", dest="with_pip", action="store_true")
    pip_group.add_argument("--no-pip", dest="with_pip", action="store_false")
    parser.set_defaults(with_pip=True)
    try:
        args = parser.parse_args(invocation.args)
    except SystemExit:
        return _usage_error(
            "Usage: create-env <name> [--description TEXT] [--json] [--no-pip|--with-pip]"
        )

    try:
        info = shell.python_envs.create(
            args.name,
            requested_by=shell.user,
            description=args.description,
            with_pip=args.with_pip,
        )
    except InvalidPythonEnvironmentName as exc:
        return CommandResult(status=1, stderr=str(exc) + "\n")
    except PythonEnvironmentExistsError as exc:
        return CommandResult(status=1, stderr=str(exc) + "\n")
    except PythonEnvironmentError as exc:
        return CommandResult(status=1, stderr=str(exc) + "\n")

    payload = info.to_dict()
    audit: Dict[str, object] = {
        "environment_id": info.environment_id,
        "with_pip": info.with_pip,
        "ledger_event_ids": payload.get("ledger_event_ids", {}),
    }
    if args.description:
        audit["description"] = args.description
    if payload.get("metadata_path"):
        audit["metadata_path"] = payload["metadata_path"]
    audit["status"] = 0

    if args.as_json:
        stdout = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    else:
        parts = [
            f"Environment {info.environment_id} created",
            f"path={payload['path']}",
            f"python={payload['python_executable']}",
        ]
        duration = payload.get("duration_ms")
        if duration is not None:
            parts.append(f"duration={float(duration):.2f}ms")
        created_event = payload.get("ledger_event_ids", {}).get("created")
        if created_event:
            parts.append(f"ledger={created_event}")
        if payload.get("metadata_path"):
            parts.append(f"metadata={payload['metadata_path']}")
        stdout = " ".join(parts) + "\n"

    return CommandResult(stdout=stdout, status=0, audit=audit)


@command(
    name="self-task-review",
    summary="Manage external AI provider registry and log deterministic task events",
    usage=(
        "self-task-review [--json] <list|enable|disable|set-key|record> ..."
    ),
    capabilities=("admin",),
)
def self_task_review_command(shell: ShellSession, invocation: CommandInvocation) -> CommandResult:
    args = list(invocation.args)
    as_json = False
    if args and args[0] == "--json":
        as_json = True
        args = args[1:]
    if not args:
        return _usage_error(
            "Usage: self-task-review [--json] <list|enable|disable|set-key|record> ..."
        )

    action = args[0]
    manager = shell.self_task_review

    def _emit(payload: Dict[str, object], *, audit: Dict[str, object]) -> CommandResult:
        if as_json:
            body = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
        else:
            providers = payload.get("providers")
            if isinstance(providers, list):
                lines = []
                for item in providers:
                    if not isinstance(item, dict):
                        continue
                    name = str(item.get("provider_name", "unknown"))
                    status = "active" if item.get("active") else "inactive"
                    connected = item.get("codex_api_connected")
                    if connected is None:
                        connected = item.get("api_connected")
                    lines.append(
                        f"{name:12s} status={status} connected={bool(connected)}"
                    )
                body = "\n".join(lines) + ("\n" if lines else "No providers registered\n")
            else:
                provider = payload.get("provider")
                if isinstance(provider, dict):
                    name = provider.get("provider_name", "unknown")
                    status = "active" if provider.get("active") else "inactive"
                    connected = provider.get("codex_api_connected")
                    if connected is None:
                        connected = provider.get("api_connected")
                    body = (
                        f"Provider {name} {status}"
                        f" (connected={bool(connected)})\n"
                    )
                else:
                    body = json.dumps(payload, ensure_ascii=False) + "\n"
        return CommandResult(stdout=body, audit=audit)

    try:
        if action == "list":
            providers = manager.list_providers()
            audit = {"registry_action": "list", "providers": len(providers)}
            return _emit({"providers": providers}, audit=audit)

        if action == "enable":
            parser = argparse.ArgumentParser(
                prog="self-task-review enable", add_help=False
            )
            parser.add_argument("provider")
            parser.add_argument("--api-key", dest="api_key")
            try:
                parsed = parser.parse_args(args[1:])
            except SystemExit:
                return _usage_error(
                    "Usage: self-task-review enable <provider> [--api-key KEY]"
                )
            result = manager.enable_provider(parsed.provider, api_key=parsed.api_key)
            ledger_event = result.get("ledger_event", {})
            audit = {
                "registry_action": "enable",
                "provider": parsed.provider,
                "ledger_event_id": ledger_event.get("event_id"),
            }
            return _emit(result, audit=audit)

        if action == "disable":
            parser = argparse.ArgumentParser(
                prog="self-task-review disable", add_help=False
            )
            parser.add_argument("provider")
            try:
                parsed = parser.parse_args(args[1:])
            except SystemExit:
                return _usage_error("Usage: self-task-review disable <provider>")
            result = manager.disable_provider(parsed.provider)
            ledger_event = result.get("ledger_event", {})
            audit = {
                "registry_action": "disable",
                "provider": parsed.provider,
                "ledger_event_id": ledger_event.get("event_id"),
            }
            return _emit(result, audit=audit)

        if action == "set-key":
            parser = argparse.ArgumentParser(
                prog="self-task-review set-key", add_help=False
            )
            parser.add_argument("provider")
            parser.add_argument("api_key")
            try:
                parsed = parser.parse_args(args[1:])
            except SystemExit:
                return _usage_error(
                    "Usage: self-task-review set-key <provider> <api_key>"
                )
            api_key = parsed.api_key
            if api_key == "-":
                api_key = ""
            result = manager.set_api_key(parsed.provider, api_key)
            ledger_event = result.get("ledger_event", {})
            audit = {
                "registry_action": "set-key",
                "provider": parsed.provider,
                "ledger_event_id": ledger_event.get("event_id"),
            }
            return _emit(result, audit=audit)

        if action == "record":
            parser = argparse.ArgumentParser(
                prog="self-task-review record", add_help=False
            )
            parser.add_argument("provider")
            parser.add_argument("task_id")
            parser.add_argument("status")
            parser.add_argument("runtime_ms", type=float)
            parser.add_argument("output_hash")
            try:
                parsed = parser.parse_args(args[1:])
            except SystemExit:
                return _usage_error(
                    "Usage: self-task-review record <provider> <task_id> <status> <runtime_ms> <output_hash>"
                )
            result = manager.record_task_event(
                parsed.provider,
                parsed.task_id,
                parsed.status,
                parsed.runtime_ms,
                parsed.output_hash,
            )
            ledger_event = result.get("ledger_event", {})
            audit = {
                "registry_action": "record",
                "provider": parsed.provider,
                "task_id": parsed.task_id,
                "ledger_event_id": ledger_event.get("event_id"),
            }
            return _emit(result, audit=audit)
    except SelfTaskReviewError as exc:
        return CommandResult(status=1, stderr=str(exc) + "\n")

    return _usage_error(
        "Usage: self-task-review [--json] <list|enable|disable|set-key|record> ..."
    )


@command(
    name="kernel-updates",
    summary="Distribute kernel updates with token-signed packages",
    usage="kernel-updates [--json] <list|distribute> ...",
    capabilities=("admin",),
)
def kernel_updates_command(shell: ShellSession, invocation: CommandInvocation) -> CommandResult:
    args = list(invocation.args)
    emit_json = False
    if args and args[0] == "--json":
        emit_json = True
        args = args[1:]
    if not args:
        return _usage_error("Usage: kernel-updates [--json] <list|distribute> ...")

    action = args[0]
    manager = shell.kernel_updates

    if action == "list":
        packages = manager.list_packages()
        audit = {"command": "kernel-updates", "action": "list", "packages": packages}
        if emit_json:
            body = json.dumps({"packages": packages}, indent=2, ensure_ascii=False)
            return CommandResult(stdout=body + "\n", audit=audit)
        if not packages:
            return CommandResult(stdout="No kernel updates distributed\n", audit=audit)
        lines = ["Distributed kernel updates:"]
        for package in packages:
            lines.append(
                "  #"
                + str(package["package_id"]).rjust(3, "0")
                + f" {package['version']} ({package['artifact']})"
            )
            lines.append(
                f"      sha256={package['sha256']} size={package['size_bytes']} bytes token={package['token_id']}"
            )
        return CommandResult(stdout="\n".join(lines) + "\n", audit=audit)

    if action == "distribute":
        parser = argparse.ArgumentParser(prog="kernel-updates distribute", add_help=False)
        parser.add_argument("version")
        parser.add_argument("artifact")
        parser.add_argument("--sha256", required=True)
        parser.add_argument("--token-id", default="default")
        parser.add_argument("--signature", required=True)
        parser.add_argument("--signature-algorithm", default="hmac-sha256")
        parser.add_argument("--notes", default="")
        try:
            parsed = parser.parse_args(args[1:])
        except SystemExit:
            return _usage_error(
                "Usage: kernel-updates [--json] distribute <version> <artifact> --sha256 DIGEST "
                "--signature SIGNATURE [--token-id TOKEN] [--signature-algorithm ALGO] [--notes TEXT]"
            )
        artifact_path = (shell.cwd / Path(parsed.artifact)).resolve()
        try:
            result = manager.distribute(
                parsed.version,
                artifact_path,
                sha256=parsed.sha256,
                token_id=parsed.token_id,
                signature=parsed.signature,
                signature_algorithm=parsed.signature_algorithm,
                notes=parsed.notes,
            )
        except KernelUpdateError as exc:
            return CommandResult(status=1, stderr=str(exc) + "\n")
        audit = {"command": "kernel-updates", "action": "distribute"}
        audit.update(result)
        if emit_json:
            body = json.dumps(result, indent=2, ensure_ascii=False)
            return CommandResult(stdout=body + "\n", audit=audit)
        package = result["package"]
        lines = [
            f"Distributed kernel update #{package['package_id']} version {package['version']}",
            f"  Artifact: {package['artifact']}",
            f"  SHA256: {package['sha256']}",
            f"  Token: {package['token_id']} (algo={package['signature_algorithm']})",
        ]
        if package.get("notes"):
            lines.append(f"  Notes: {package['notes']}")
        return CommandResult(stdout="\n".join(lines) + "\n", audit=audit)

    return _usage_error("Usage: kernel-updates [--json] <list|distribute> ...")


@command(
    name="kernel-performance",
    summary="Report AI kernel energy, memory, and I/O metrics",
    usage="kernel-performance [--json] <summary|list|record> ...",
    capabilities=("admin",),
)
def kernel_performance_command(
    shell: ShellSession, invocation: CommandInvocation
) -> CommandResult:
    args = list(invocation.args)
    emit_json = False
    if args and args[0] == "--json":
        emit_json = True
        args = args[1:]
    if not args:
        return _usage_error(
            "Usage: kernel-performance [--json] <summary|list|record> ..."
        )

    action = args[0]
    monitor = shell.kernel_performance

    def _json(payload: Dict[str, object]) -> str:
        return json.dumps(payload, indent=2, ensure_ascii=False) + "\n"

    try:
        if action == "summary":
            summary = monitor.summary()
            audit = {
                "command": "kernel-performance",
                "action": "summary",
                "summary": summary,
            }
            if emit_json:
                return CommandResult(stdout=_json({"summary": summary}), audit=audit)
            lines = [
                "AI kernel performance summary:",
                f"  Samples: {summary['samples']}",
                (
                    "  Total energy: "
                    f"{summary['total_energy_joules']:.3f} J"
                    f" (avg {summary['average_energy_joules']:.3f} J)"
                ),
                (
                    "  Total memory: "
                    f"{summary['total_memory_kb']} KB"
                    f" (avg {summary['average_memory_kb']:.1f} KB)"
                ),
                (
                    "  Total I/O: "
                    f"{summary['total_io_bytes']} bytes"
                    f" (avg {summary['average_io_bytes']:.1f} bytes)"
                ),
                f"  Peak memory: {summary['peak_memory_kb']} KB",
                f"  Peak I/O: {summary['peak_io_bytes']} bytes",
            ]
            energy_per_io = summary.get("energy_per_io_byte", 0.0) or 0.0
            if energy_per_io:
                lines.append(
                    f"  Energy per I/O byte: {energy_per_io:.6f} J"
                )
            last_recorded = summary.get("last_recorded_at")
            if last_recorded:
                lines.append(f"  Last sample at {last_recorded}")
            components = summary.get("components", {})
            if components:
                lines.append("Component breakdown:")
                for name in sorted(components):
                    component = components[name]
                    lines.append(
                        "  - "
                        + name
                        + ": samples="
                        + str(component.get("samples", 0))
                        + f" energy={component.get('total_energy_joules', 0.0):.3f}J"
                        + f" peak_memory={component.get('peak_memory_kb', 0)}KB"
                        + f" io={component.get('total_io_bytes', 0)}B"
                    )
            return CommandResult(stdout="\n".join(lines) + "\n", audit=audit)

        if action == "list":
            samples = monitor.list()
            audit = {
                "command": "kernel-performance",
                "action": "list",
                "samples": samples,
            }
            if emit_json:
                return CommandResult(stdout=_json({"samples": samples}), audit=audit)
            if not samples:
                return CommandResult(
                    stdout="No performance samples recorded\n",
                    audit=audit,
                )
            lines = ["Recorded kernel performance samples:"]
            for sample in samples:
                sample_id = int(sample.get("sample_id", 0))
                recorded_at = sample.get("recorded_at", "unknown")
                component = sample.get("component", "kernel")
                energy = float(sample.get("energy_joules", 0.0))
                memory = int(sample.get("memory_kb", 0))
                io_bytes = int(sample.get("io_bytes", 0))
                lines.append(
                    f"  #{sample_id:03d} {recorded_at} component={component}"
                )
                lines.append(
                    f"      energy={energy:.3f}J memory={memory}KB io={io_bytes}B"
                )
                notes = sample.get("notes")
                if notes:
                    lines.append(f"      notes: {notes}")
            return CommandResult(stdout="\n".join(lines) + "\n", audit=audit)

        if action == "record":
            parser = argparse.ArgumentParser(
                prog="kernel-performance record", add_help=False
            )
            parser.add_argument("energy_joules", type=float)
            parser.add_argument("memory_kb", type=int)
            parser.add_argument("io_bytes", type=int)
            parser.add_argument("--component", default="kernel")
            parser.add_argument("--notes", default="")
            try:
                parsed = parser.parse_args(args[1:])
            except SystemExit:
                return _usage_error(
                    "Usage: kernel-performance [--json] record <energy_joules> <memory_kb> <io_bytes> [--component NAME] [--notes TEXT]"
                )
            result = monitor.record(
                parsed.energy_joules,
                parsed.memory_kb,
                parsed.io_bytes,
                component=parsed.component,
                notes=parsed.notes,
            )
            audit = {
                "command": "kernel-performance",
                "action": "record",
                "sample": result.get("sample"),
                "ledger_event_id": (result.get("ledger_event") or {}).get(
                    "event_id"
                ),
            }
            if emit_json:
                return CommandResult(stdout=_json(result), audit=audit)
            sample = result["sample"]
            lines = [
                (
                    "Recorded performance sample #"
                    + str(sample["sample_id"])
                    + f" ({sample['component']}):"
                ),
                (
                    "  energy="
                    + f"{float(sample['energy_joules']):.3f}J"
                    + f" memory={sample['memory_kb']}KB"
                    + f" io={sample['io_bytes']}B"
                ),
            ]
            notes = sample.get("notes")
            if notes:
                lines.append(f"  notes: {notes}")
            return CommandResult(stdout="\n".join(lines) + "\n", audit=audit)
    except KernelPerformanceError as exc:
        return CommandResult(status=1, stderr=str(exc) + "\n")

    return _usage_error(
        "Usage: kernel-performance [--json] <summary|list|record> ..."
    )


@command(
    name="gpu-access",
    summary="Acquire or release secure GPU access leases",
    usage="gpu-access [--json] <list|acquire|release> ...",
    capabilities=("admin",),
)
def gpu_access_command(shell: ShellSession, invocation: CommandInvocation) -> CommandResult:
    args = list(invocation.args)
    output_json = False
    if args and args[0] == "--json":
        output_json = True
        args = args[1:]
    if not args:
        return _usage_error("Usage: gpu-access [--json] <list|acquire|release> ...")

    action = args[0]
    manager = shell.gpu_manager

    if action == "list":
        leases = manager.list_leases()
        audit = {"leases": leases}
        if output_json:
            return CommandResult(stdout=json.dumps({"leases": leases}, indent=2) + "\n", audit=audit)
        if not leases:
            return CommandResult(stdout="No active GPU leases\n", audit=audit)
        lines = ["Active GPU leases:"]
        for lease_id, metadata in leases.items():
            lines.append(
                f"  {lease_id}: capability={metadata['capability']} backend={metadata['backend']} device={metadata['device']}"
            )
        return CommandResult(stdout="\n".join(lines) + "\n", audit=audit)

    if action == "acquire":
        parser = argparse.ArgumentParser(prog="gpu-access acquire", add_help=False)
        parser.add_argument("capability")
        parser.add_argument("backend")
        parser.add_argument("device")
        parser.add_argument("--model")
        try:
            parsed = parser.parse_args(args[1:])
        except SystemExit:
            return _usage_error("Usage: gpu-access [--json] acquire <capability> <backend> <device> [--model NAME]")
        try:
            lease = manager.acquire(
                capability=parsed.capability,
                backend=parsed.backend,
                device=parsed.device,
                model=parsed.model,
            )
        except GPUAccessError as exc:
            return CommandResult(status=1, stderr=str(exc) + "\n")
        audit = {
            "lease_id": lease.lease_id,
            "capability": lease.capability,
            "backend": lease.backend,
            "device": lease.device,
        }
        if output_json:
            payload = {
                "lease_id": lease.lease_id,
                "capability": lease.capability,
                "backend": lease.backend,
                "device": lease.device,
                "granted_event": lease.granted_event,
            }
            return CommandResult(stdout=json.dumps(payload, indent=2) + "\n", audit=audit)
        lines = [
            f"Lease granted {lease.lease_id}",
            f"  capability={lease.capability}",
            f"  backend={lease.backend}",
            f"  device={lease.device}",
        ]
        return CommandResult(stdout="\n".join(lines) + "\n", audit=audit)

    if action == "release":
        parser = argparse.ArgumentParser(prog="gpu-access release", add_help=False)
        parser.add_argument("lease_id")
        parser.add_argument("--status", default="completed")
        parser.add_argument("--tokens", type=int, default=0)
        try:
            parsed = parser.parse_args(args[1:])
        except SystemExit:
            return _usage_error("Usage: gpu-access [--json] release <lease_id> [--status STATUS] [--tokens N]")
        try:
            event = manager.release(
                parsed.lease_id,
                status=parsed.status,
                tokens=parsed.tokens,
            )
        except GPUAccessError as exc:
            return CommandResult(status=1, stderr=str(exc) + "\n")
        audit = {
            "lease_id": parsed.lease_id,
            "status": parsed.status,
            "tokens": parsed.tokens,
            "ledger_event_id": event.get("event_id"),
        }
        if output_json:
            return CommandResult(stdout=json.dumps(audit, indent=2) + "\n", audit=audit)
        return CommandResult(stdout=f"Lease {parsed.lease_id} released\n", audit=audit)

    return _usage_error("Usage: gpu-access [--json] <list|acquire|release> ...")


@command(
    name="entropy-audit",
    summary="Audit entropy events for deviations",
    usage="entropy-audit [--json] [--limit N]",
    capabilities=("admin",),
)
def entropy_audit_command(shell: ShellSession, invocation: CommandInvocation) -> CommandResult:
    parser = argparse.ArgumentParser(prog="entropy-audit", add_help=False)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--limit", type=int)
    try:
        parsed = parser.parse_args(invocation.args)
    except SystemExit:
        return _usage_error("Usage: entropy-audit [--json] [--limit N]")
    try:
        result = shell.entropy_auditor.audit(limit=parsed.limit)
    except SnapshotLedgerError as exc:
        return CommandResult(status=1, stderr=str(exc) + "\n")
    audit = {
        "total_events": result.get("total_events", 0),
        "deviations": result.get("deviations", []),
    }
    if parsed.json:
        return CommandResult(stdout=json.dumps(result, indent=2) + "\n", audit=audit)
    lines = [
        f"Entropy events audited: {result.get('total_events', 0)}",
        f"Deviations detected: {len(result.get('deviations', []))}",
    ]
    for deviation in result.get("deviations", []):
        lines.append(
            "  - "
            + deviation.get("event_id", "unknown")
            + f" bits={deviation.get('entropy_bits')}"
        )
    return CommandResult(stdout="\n".join(lines) + "\n", audit=audit)


@command(
    name="integrity-check",
    summary="Run integrity hash checks across critical files",
    usage="integrity-check [--json] [--label NAME] [paths...]",
    capabilities=("admin",),
)
def integrity_check_command(shell: ShellSession, invocation: CommandInvocation) -> CommandResult:
    parser = argparse.ArgumentParser(prog="integrity-check", add_help=False)
    parser.add_argument("--label", default="integrity")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("paths", nargs="*")
    try:
        parsed = parser.parse_args(invocation.args)
    except SystemExit:
        return _usage_error("Usage: integrity-check [--json] [--label NAME] [paths...]")
    if parsed.paths:
        paths = [(shell.cwd / Path(item)).resolve() for item in parsed.paths]
    else:
        paths = [
            shell.root / "cli" / "data" / "snapshot_ledger.jsonl",
            shell.root / "cli" / "python_vm" / "sessions",
        ]
    report = shell.integrity_monitor.run_check(label=parsed.label, paths=paths)
    audit = {
        "label": report.label,
        "digest": report.digest,
        "files_hashed": report.files_hashed,
    }
    if parsed.json:
        return CommandResult(stdout=json.dumps(audit, indent=2) + "\n", audit=audit)
    lines = [
        f"Integrity check {report.label}",
        f"  files_hashed={report.files_hashed}",
        f"  digest={report.digest}",
    ]
    return CommandResult(stdout="\n".join(lines) + "\n", audit=audit)


@command(
    name="secure-backup",
    summary="Create signed backups outside the kernel workspace",
    usage="secure-backup [--json] create <label> [--token TOKEN] [paths...]",
    capabilities=("admin",),
)
def secure_backup_command(shell: ShellSession, invocation: CommandInvocation) -> CommandResult:
    args = list(invocation.args)
    output_json = False
    if args and args[0] == "--json":
        output_json = True
        args = args[1:]
    if not args or args[0] != "create":
        return _usage_error("Usage: secure-backup [--json] create <label> [--token TOKEN] [paths...]")
    parser = argparse.ArgumentParser(prog="secure-backup create", add_help=False)
    parser.add_argument("label")
    parser.add_argument("paths", nargs="*")
    parser.add_argument("--token", default="default")
    try:
        parsed = parser.parse_args(args[1:])
    except SystemExit:
        return _usage_error("Usage: secure-backup [--json] create <label> [--token TOKEN] [paths...]")
    if parsed.paths:
        include = [(shell.cwd / Path(item)).resolve() for item in parsed.paths]
    else:
        include = [
            shell.root / "cli" / "data" / "snapshot_ledger.jsonl",
            shell.root / "cli" / "data" / "download_ledger.jsonl",
        ]
    summary = shell.backup_manager.create_backup(
        label=parsed.label,
        include=include,
        token_id=parsed.token,
    )
    audit = {
        "label": summary.backup_dir.name,
        "digest": summary.digest,
        "signature": summary.signature,
        "token_id": summary.token_id,
    }
    if output_json:
        payload = {
            "backup_dir": str(summary.backup_dir),
            "digest": summary.digest,
            "signature": summary.signature,
            "token_id": summary.token_id,
            "files": [str(path) for path in summary.files],
        }
        return CommandResult(stdout=json.dumps(payload, indent=2) + "\n", audit=audit)
    lines = [
        f"Backup stored at {summary.backup_dir}",
        f"  digest={summary.digest}",
        f"  signature={summary.signature}",
    ]
    return CommandResult(stdout="\n".join(lines) + "\n", audit=audit)


@command(
    name="security-log",
    summary="Record and integrate security events",
    usage="security-log [--json] <list|record|integrate> ...",
    capabilities=("admin",),
)
def security_log_command(shell: ShellSession, invocation: CommandInvocation) -> CommandResult:
    args = list(invocation.args)
    output_json = False
    if args and args[0] == "--json":
        output_json = True
        args = args[1:]
    if not args:
        return _usage_error("Usage: security-log [--json] <list|record|integrate> ...")

    action = args[0]
    manager = shell.security_log

    if action == "list":
        events = list(manager.iter_events() or [])
        payload = [event.__dict__ for event in events]
        audit = {"events": payload}
        if output_json:
            return CommandResult(stdout=json.dumps({"events": payload}, indent=2) + "\n", audit=audit)
        if not events:
            return CommandResult(stdout="No security events recorded\n", audit=audit)
        lines = ["Security events:"]
        for event in events:
            lines.append(
                f"  {event.event_id}: {event.event_type} severity={event.severity}"
            )
        return CommandResult(stdout="\n".join(lines) + "\n", audit=audit)

    if action == "record":
        parser = argparse.ArgumentParser(prog="security-log record", add_help=False)
        parser.add_argument("event_type")
        parser.add_argument("message")
        parser.add_argument("--severity", default="info")
        try:
            parsed = parser.parse_args(args[1:])
        except SystemExit:
            return _usage_error("Usage: security-log [--json] record <event_type> <message> [--severity LEVEL]")
        event = manager.record(
            event_type=parsed.event_type,
            message=parsed.message,
            severity=parsed.severity,
        )
        audit = event.__dict__
        if output_json:
            return CommandResult(stdout=json.dumps(audit, indent=2) + "\n", audit=audit)
        return CommandResult(stdout=f"Recorded security event {event.event_id}\n", audit=audit)

    if action == "integrate":
        result = manager.integrate_with_replay(shell.ai_replay)
        audit = dict(result)
        if output_json:
            return CommandResult(stdout=json.dumps(result, indent=2) + "\n", audit=audit)
        return CommandResult(
            stdout=f"Integrated {result['count']} security events into replay storage\n",
            audit=audit,
        )

    return _usage_error("Usage: security-log [--json] <list|record|integrate> ...")


@command(
    name="snapshot-benchmarks",
    summary="Evaluate system behavior with periodic snapshot benchmarks",
    usage="snapshot-benchmarks [--json] <status|run> [--force]",
    capabilities=("admin",),
)
def snapshot_benchmarks_command(
    shell: ShellSession, invocation: CommandInvocation
) -> CommandResult:
    args = list(invocation.args)
    emit_json = False
    if args and args[0] == "--json":
        emit_json = True
        args = args[1:]
    if not args:
        return _usage_error(
            "Usage: snapshot-benchmarks [--json] <status|run> [--force]"
        )

    action = args[0]
    manager = shell.snapshot_benchmarks

    if action == "status":
        status_payload = manager.status()
        audit = {"command": "snapshot-benchmarks", "action": "status", **status_payload}
        if emit_json:
            body = json.dumps(status_payload, indent=2, ensure_ascii=False)
            return CommandResult(stdout=body + "\n", audit=audit)
        last = status_payload.get("last_benchmark")
        if not last:
            return CommandResult(
                stdout="No snapshot benchmarks recorded\n",
                audit=audit,
            )
        summary = last.get("task_summary", {})
        success_rate = summary.get("success_rate", 0.0)
        total = summary.get("total", 0)
        attempted = summary.get("attempted", 0)
        success = summary.get("success", 0)
        failure = summary.get("failure", 0)
        duration_ms = last.get("duration_ms", 0.0)
        lines = [
            f"Last snapshot benchmark #{last['benchmark_id']} completed at {last['completed_at']}",
            f"  Duration: {duration_ms:.2f} ms",
            f"  Tasks: total={total} attempted={attempted} success={success} failure={failure}",
            f"  Success rate: {success_rate:.2%}",
        ]
        next_allowed = status_payload.get("next_allowed_at")
        remaining = float(status_payload.get("seconds_until_next", 0.0) or 0.0)
        if next_allowed:
            if remaining > 0:
                hours = int(remaining // 3600)
                minutes = int((remaining % 3600) // 60)
                lines.append(
                    "Next benchmark allowed at "
                    f"{next_allowed} (~{hours}h {minutes}m remaining)"
                )
            else:
                lines.append(f"Next benchmark allowed at {next_allowed} (available now)")
        else:
            lines.append("Next benchmark may run immediately")
        return CommandResult(stdout="\n".join(lines) + "\n", audit=audit)

    if action == "run":
        parser = argparse.ArgumentParser(prog="snapshot-benchmarks run", add_help=False)
        parser.add_argument("--force", action="store_true")
        try:
            parsed = parser.parse_args(args[1:])
        except SystemExit:
            return _usage_error(
                "Usage: snapshot-benchmarks [--json] run [--force]"
            )
        try:
            result = manager.evaluate(force=parsed.force)
        except SnapshotBenchmarkError as exc:
            return CommandResult(status=1, stderr=str(exc) + "\n")
        audit = {"command": "snapshot-benchmarks", "action": "run", **result}
        if emit_json:
            body = json.dumps(result, indent=2, ensure_ascii=False)
            return CommandResult(stdout=body + "\n", audit=audit)
        benchmark = result["benchmark"]
        summary = benchmark.get("task_summary", {})
        lines = [
            f"Recorded snapshot benchmark #{benchmark['benchmark_id']}",
            f"  Started: {benchmark['started_at']}",
            f"  Completed: {benchmark['completed_at']}",
            f"  Duration: {benchmark['duration_ms']:.2f} ms",
            f"  Success rate: {summary.get('success_rate', 0.0):.2%}",
        ]
        return CommandResult(stdout="\n".join(lines) + "\n", audit=audit)

    return _usage_error(
        "Usage: snapshot-benchmarks [--json] <status|run> [--force]"
    )


@command(
    name="deterministic-benchmark",
    summary="Run the deterministic validation suite and inspect prior runs",
    usage="deterministic-benchmark [--json] <status|run> [options]",
    capabilities=("admin",),
)
def deterministic_benchmark_command(
    shell: ShellSession, invocation: CommandInvocation
) -> CommandResult:
    args = list(invocation.args)
    emit_json = False
    if args and args[0] == "--json":
        emit_json = True
        args = args[1:]
    if not args:
        return _usage_error(
            "Usage: deterministic-benchmark [--json] <status|run> [options]"
        )

    action = args[0]

    if action == "status":
        status = shell.deterministic_benchmark.status()
        if not status:
            message = "No deterministic benchmark runs recorded\n"
            return CommandResult(stdout=message, audit={"runs": 0})
        audit = {"command": "deterministic-benchmark", "action": "status", **status}
        if emit_json:
            body = json.dumps(status, indent=2, ensure_ascii=False)
            return CommandResult(stdout=body + "\n", audit=audit)
        python_result = status.get("python_sessions", {})
        stress_result = status.get("stress", {})
        replay_result = status.get("replay_consistency", {})
        lines = [
            f"Deterministic benchmark run #{status.get('run_id')} at {status.get('executed_at')}",
            f"  python sessions: {python_result.get('verified_sessions', 0)}/{python_result.get('checked_sessions', 0)}",
            f"  stress iterations: {stress_result.get('iterations', 0)} avg_energy={stress_result.get('average_energy_joules', 0.0):.3f}",
            f"  replay digest: {replay_result.get('final_digest', 'n/a')}",
        ]
        ai_graph = status.get("ai_report", {}).get("graph")
        if ai_graph:
            lines.append("  AI graph:")
            for line in str(ai_graph).splitlines():
                lines.append(f"    {line}")
        return CommandResult(stdout="\n".join(lines) + "\n", audit=audit)

    if action == "run":
        parser = argparse.ArgumentParser(
            prog="deterministic-benchmark run", add_help=False
        )
        parser.add_argument("--count", type=int, default=1000)
        parser.add_argument("--stress-iterations", type=int, default=25)
        parser.add_argument("--limit", type=int, default=None)
        try:
            parsed = parser.parse_args(args[1:])
        except SystemExit:
            return _usage_error(
                "Usage: deterministic-benchmark [--json] run [--count N] "
                "[--stress-iterations N] [--limit N]"
            )
        try:
            result = shell.deterministic_benchmark.run_suite(
                replay_count=max(1, parsed.count),
                stress_iterations=max(1, parsed.stress_iterations),
                python_limit=parsed.limit,
            )
        except DeterministicBenchmarkError as exc:
            return CommandResult(status=1, stderr=str(exc) + "\n")

        audit = {
            "command": "deterministic-benchmark",
            "action": "run",
            "run_id": result.get("run_id"),
            "python_sessions": result.get("python_sessions", {}).get(
                "verified_sessions", 0
            ),
            "replay_count": result.get("replay_consistency", {}).get("count", 0),
            "stress_iterations": result.get("stress", {}).get("iterations", 0),
            "build_digest": result.get("build_signature", {}).get("digest"),
        }

        if emit_json:
            body = json.dumps(result, indent=2, ensure_ascii=False)
            return CommandResult(stdout=body + "\n", audit=audit)

        python_result = result.get("python_sessions", {})
        stress_result = result.get("stress", {})
        replay_result = result.get("replay_consistency", {})
        export_result = result.get("journey_export", {})
        build = result.get("build_signature", {})
        lines = [
            f"Deterministic benchmark run #{result.get('run_id')} completed",
            f"  executed_at: {result.get('executed_at')}",
            f"  python sessions: {python_result.get('verified_sessions', 0)}/{python_result.get('checked_sessions', 0)}",
            f"  stress iterations: {stress_result.get('iterations', 0)} avg_energy={stress_result.get('average_energy_joules', 0.0):.3f}",
            f"  replay digest: {replay_result.get('final_digest', 'n/a')}",
            f"  journey export: {export_result.get('path', 'n/a')} (tasks={export_result.get('tasks_total', 0)})",
            f"  build signature: {build.get('signature', 'n/a')}",
        ]
        ai_graph = result.get("ai_report", {}).get("graph")
        if ai_graph:
            lines.append("  AI graph:")
            for line in str(ai_graph).splitlines():
                lines.append(f"    {line}")
        return CommandResult(stdout="\n".join(lines) + "\n", audit=audit)

    return _usage_error(
        "Usage: deterministic-benchmark [--json] <status|run> [options]"
    )


@command(
    name="deterministic-recompile",
    summary="Approve code changes via deterministic recompilation",
    usage="deterministic-recompile [--json] <queue|pending|approve|history> ...",
    capabilities=("admin",),
)
def deterministic_recompile_command(
    shell: ShellSession, invocation: CommandInvocation
) -> CommandResult:
    args = list(invocation.args)
    emit_json = False
    if args and args[0] == "--json":
        emit_json = True
        args = args[1:]
    if not args:
        return _usage_error(
            "Usage: deterministic-recompile [--json] <queue|pending|approve|history> ..."
        )

    action = args[0]
    manager = shell.deterministic_recompile
    audit: Dict[str, Any] = {
        "command": "deterministic-recompile",
        "action": action,
    }

    if action == "pending":
        pending = manager.pending()
        audit["pending"] = pending
        if emit_json:
            payload = json.dumps({"pending": pending}, indent=2, ensure_ascii=False)
            return CommandResult(stdout=payload + "\n", audit=audit)
        if not pending:
            return CommandResult(
                stdout="No deterministic changes awaiting approval\n",
                audit=audit,
            )
        lines = ["Pending deterministic changes:"]
        for entry in pending:
            lines.append(
                "  "
                f"#{entry['submission_id']:03d} {entry['change_id']} "
                f"({len(entry['paths'])} files)"
            )
            if entry.get("description"):
                lines.append(f"      {entry['description']}")
            if entry.get("tags"):
                lines.append(f"      tags: {', '.join(entry['tags'])}")
            outcome = entry.get("last_outcome")
            if outcome:
                lines.append(
                    f"      last outcome: {outcome.get('status')}"
                )
        return CommandResult(stdout="\n".join(lines) + "\n", audit=audit)

    if action == "history":
        parser = argparse.ArgumentParser(
            prog="deterministic-recompile history", add_help=False
        )
        parser.add_argument("--limit", type=int, default=None)
        try:
            parsed = parser.parse_args(args[1:])
        except SystemExit:
            return _usage_error(
                "Usage: deterministic-recompile [--json] history [--limit N]"
            )
        records = manager.history(limit=parsed.limit)
        audit["history"] = records
        if emit_json:
            payload = json.dumps({"history": records}, indent=2, ensure_ascii=False)
            return CommandResult(stdout=payload + "\n", audit=audit)
        if not records:
            return CommandResult(
                stdout="No deterministic approvals recorded yet\n",
                audit=audit,
            )
        lines = ["Deterministic approval history:"]
        for entry in records:
            lines.append(
                "  "
                f"#{entry['submission_id']:03d} {entry['change_id']} "
                f"approved {entry['approved_at']} by {entry.get('reviewer', 'unknown')}"
            )
            if entry.get("description"):
                lines.append(f"      {entry['description']}")
        return CommandResult(stdout="\n".join(lines) + "\n", audit=audit)

    if action == "queue":
        parser = argparse.ArgumentParser(
            prog="deterministic-recompile queue", add_help=False
        )
        parser.add_argument("change_id")
        parser.add_argument("--path", dest="paths", action="append", required=True)
        parser.add_argument("--description", default="")
        parser.add_argument("--tag", dest="tags", action="append", default=[])
        try:
            parsed = parser.parse_args(args[1:])
        except SystemExit:
            return _usage_error(
                "Usage: deterministic-recompile [--json] queue <change-id> --path FILE [--path FILE] [--description TEXT] [--tag TAG]..."
            )
        try:
            result = manager.queue(
                parsed.change_id,
                parsed.paths,
                description=parsed.description,
                submitted_by=shell.user,
                tags=parsed.tags,
            )
        except DeterministicRecompileError as exc:
            return CommandResult(status=1, stderr=str(exc) + "\n")
        audit.update(result)
        queued = result["queued_change"]
        if emit_json:
            payload = json.dumps(result, indent=2, ensure_ascii=False)
            return CommandResult(stdout=payload + "\n", audit=audit)
        lines = [
            f"Queued change {queued['change_id']} for deterministic approval",
            f"  Submission ID: {queued['submission_id']}",
            f"  Paths: {', '.join(queued['paths'])}",
        ]
        if queued.get("description"):
            lines.append(f"  Description: {queued['description']}")
        if queued.get("tags"):
            lines.append(f"  Tags: {', '.join(queued['tags'])}")
        return CommandResult(stdout="\n".join(lines) + "\n", audit=audit)

    if action == "approve":
        parser = argparse.ArgumentParser(
            prog="deterministic-recompile approve", add_help=False
        )
        parser.add_argument("change_id")
        try:
            parsed = parser.parse_args(args[1:])
        except SystemExit:
            return _usage_error(
                "Usage: deterministic-recompile [--json] approve <change-id>"
            )
        try:
            result = manager.approve(parsed.change_id, reviewer=shell.user)
        except DeterministicRecompileError as exc:
            return CommandResult(status=1, stderr=str(exc) + "\n")
        audit.update(result)
        approved = result["approved_change"]
        if emit_json:
            payload = json.dumps(result, indent=2, ensure_ascii=False)
            return CommandResult(stdout=payload + "\n", audit=audit)
        lines = [
            f"Approved change {approved['change_id']} deterministically",
            f"  Submission ID: {approved['submission_id']}",
            f"  Approved at: {approved['approved_at']}",
        ]
        return CommandResult(stdout="\n".join(lines) + "\n", audit=audit)

    return _usage_error(
        "Usage: deterministic-recompile [--json] <queue|pending|approve|history> ..."
    )


@command(
    name="self-feedback",
    summary="Analyze recent user interaction transcripts",
    usage="self-feedback [--json] <summary|recent> [--limit N]",
    capabilities=("admin",),
)
def self_feedback_command(shell: ShellSession, invocation: CommandInvocation) -> CommandResult:
    args = list(invocation.args)
    emit_json = False
    if args and args[0] == "--json":
        emit_json = True
        args = args[1:]
    if not args:
        return _usage_error(
            "Usage: self-feedback [--json] <summary|recent> [--limit N]"
        )

    action = args[0]
    analyzer = shell.self_feedback
    audit: Dict[str, Any] = {"command": "self-feedback", "action": action}

    if action == "summary":
        summary = analyzer.summary()
        audit["summary"] = summary
        if emit_json:
            payload = json.dumps({"summary": summary}, indent=2, ensure_ascii=False)
            return CommandResult(stdout=payload + "\n", audit=audit)
        lines = [
            "Self-feedback summary:",
            f"  Total interactions: {summary['total_interactions']}",
            f"  Success rate: {summary['success_rate'] * 100:.1f}%",
            f"  Positive rate: {summary['positive_rate'] * 100:.1f}%",
            f"  Error rate: {summary['error_rate'] * 100:.1f}%",
            f"  Sentiment score: {summary['sentiment_score']:.2f}",
            f"  Engagement score: {summary['engagement_score']}",
        ]
        if summary.get("last_interaction_at"):
            lines.append(f"  Last interaction: {summary['last_interaction_at']}")
        if summary.get("top_commands"):
            lines.append("  Top commands:")
            for entry in summary["top_commands"]:
                lines.append(
                    "    "
                    f"{entry['command']}: {entry['count']} runs "
                    f"(success {entry['success_rate'] * 100:.1f}%)"
                )
        if summary.get("high_friction_commands"):
            lines.append("  High friction commands:")
            for entry in summary["high_friction_commands"]:
                lines.append(
                    "    "
                    f"{entry['command']}: {entry['friction_events']} friction events "
                    f"across {entry['count']} runs"
                )
        return CommandResult(stdout="\n".join(lines) + "\n", audit=audit)

    if action == "recent":
        parser = argparse.ArgumentParser(prog="self-feedback recent", add_help=False)
        parser.add_argument("--limit", type=int, default=10)
        try:
            parsed = parser.parse_args(args[1:])
        except SystemExit:
            return _usage_error(
                "Usage: self-feedback [--json] recent [--limit N]"
            )
        interactions = analyzer.recent_interactions(parsed.limit)
        audit["interactions"] = interactions
        if emit_json:
            payload = json.dumps(
                {"interactions": interactions}, indent=2, ensure_ascii=False
            )
            return CommandResult(stdout=payload + "\n", audit=audit)
        if not interactions:
            return CommandResult(stdout="No interactions recorded yet\n", audit=audit)
        lines = ["Recent interactions:"]
        for entry in interactions:
            lines.append(
                "  "
                f"{entry['timestamp']} "
                f"{entry['command']} status={entry['status']} "
                f"sentiment={entry['sentiment']} friction={entry['friction_score']}"
            )
        return CommandResult(stdout="\n".join(lines) + "\n", audit=audit)

    return _usage_error(
        "Usage: self-feedback [--json] <summary|recent> [--limit N]"
    )


@command(
    name="task-proposals",
    summary="Allow Roken Assembly to register roadmap task proposals",
    usage="task-proposals [--json] <list|propose> ...",
    capabilities=("admin",),
)
def task_proposals_command(shell: ShellSession, invocation: CommandInvocation) -> CommandResult:
    args = list(invocation.args)
    emit_json = False
    if args and args[0] == "--json":
        emit_json = True
        args = args[1:]
    if not args:
        return _usage_error("Usage: task-proposals [--json] <list|propose> ...")

    action = args[0]
    manager = shell.task_proposals
    audit: Dict[str, Any] = {"command": "task-proposals", "action": action}

    if action == "list":
        proposals = manager.list()
        audit["proposals"] = proposals
        if emit_json:
            payload = json.dumps({"proposals": proposals}, indent=2, ensure_ascii=False)
            return CommandResult(stdout=payload + "\n", audit=audit)
        if not proposals:
            return CommandResult(stdout="No task proposals recorded\n", audit=audit)
        lines = ["Recorded task proposals:"]
        for entry in proposals:
            line = f"  #{entry['proposal_id']:03d} {entry['title']} (source={entry['source']})"
            lines.append(line)
            if entry.get("description"):
                lines.append(f"      {entry['description']}")
            if entry.get("tags"):
                lines.append(f"      tags: {', '.join(entry['tags'])}")
        return CommandResult(stdout="\n".join(lines) + "\n", audit=audit)

    if action == "propose":
        parser = argparse.ArgumentParser(prog="task-proposals propose", add_help=False)
        parser.add_argument("source")
        parser.add_argument("title")
        parser.add_argument("--description", default="")
        parser.add_argument("--tag", dest="tags", action="append", default=[])
        try:
            parsed = parser.parse_args(args[1:])
        except SystemExit:
            return _usage_error(
                "Usage: task-proposals [--json] propose <source> <title> [--description TEXT] [--tag TAG]..."
            )
        try:
            result = manager.register(
                parsed.source,
                parsed.title,
                description=parsed.description,
                tags=parsed.tags,
            )
        except TaskProposalError as exc:
            return CommandResult(status=1, stderr=str(exc) + "\n")
        audit.update(result)
        if emit_json:
            payload = json.dumps(result, indent=2, ensure_ascii=False)
            return CommandResult(stdout=payload + "\n", audit=audit)
        proposal = result["proposal"]
        lines = [
            f"Registered proposal #{proposal['proposal_id']} from {proposal['source']}:",
            f"  {proposal['title']}",
        ]
        if proposal.get("description"):
            lines.append(f"  Description: {proposal['description']}")
        if proposal.get("tags"):
            lines.append(f"  Tags: {', '.join(proposal['tags'])}")
        return CommandResult(stdout="\n".join(lines) + "\n", audit=audit)

    return _usage_error("Usage: task-proposals [--json] <list|propose> ...")


@command(
    name="pyvm",
    summary="Execute code inside the embedded Python VM",
    usage="pyvm [--resume ID] [--safe] [--token-budget N] [--eval CODE | --module MOD | script.py [args...]] [--json]",
    capabilities=("process",),
)
def pyvm(shell: ShellSession, invocation: CommandInvocation) -> CommandResult:
    return _run_python_vm_command(
        shell,
        invocation,
        usage="pyvm [--resume ID] [--safe] [--eval CODE | --module MOD | script.py [args...]] [--json]",
        allow_dash_c=False,
    )


@command(
    name="python",
    summary="Route the python alias into the embedded interpreter",
    usage="python [--resume ID] [--safe] [--token-budget N] [-c CODE | -m MODULE | script.py [args...]] [--json]",
    capabilities=("process",),
)
def python_command(shell: ShellSession, invocation: CommandInvocation) -> CommandResult:
    return _run_python_vm_command(
        shell,
        invocation,
        usage="python [--resume ID] [--safe] [-c CODE | -m MODULE | script.py [args...]] [--json]",
        allow_dash_c=True,
    )


@command(
    name="pyx",
    summary="Route the pyx alias into the embedded interpreter",
    usage="pyx [--resume ID] [--safe] [--token-budget N] [--eval CODE | --module MOD | script.py [args...]] [--json]",
    capabilities=("process",),
)
def pyx_command(shell: ShellSession, invocation: CommandInvocation) -> CommandResult:
    return _run_python_vm_command(
        shell,
        invocation,
        usage="pyx [--resume ID] [--safe] [--eval CODE | --module MOD | script.py [args...]] [--json]",
        allow_dash_c=True,
    )


@command(
    name="pip",
    summary="Invoke pip through the embedded Python interpreter",
    usage="pip [--resume ID] [--safe] [--token-budget N] [pip-args...]",
    capabilities=("process",),
)
def pip_command(shell: ShellSession, invocation: CommandInvocation) -> CommandResult:
    forwarded = dataclasses.replace(
        invocation,
        args=["-m", "pip", *invocation.args],
    )
    return _run_python_vm_command(
        shell,
        forwarded,
        usage="pip [--resume ID] [--safe] [pip-args...]",
        allow_dash_c=True,
    )


@command(
    name="git",
    summary="Execute Git commands within the workspace",
    usage="git <args...>",
    capabilities=("filesystem", "process"),
)
def git_command(shell: ShellSession, invocation: CommandInvocation) -> CommandResult:
    if not invocation.args:
        return _usage_error("Usage: git <args...>")
    if invocation.args[0] == "clone":
        return _git_clone(shell, invocation.args[1:], invocation=invocation)
    return _git_run(shell, invocation.args, invocation=invocation)


@command(
    name="audit-diff",
    summary="Compare two snapshot ledger events",
    usage="audit-diff <left-event-id> <right-event-id> [--context N] [--json]",
    capabilities=("process",),
)
def audit_diff_command(shell: ShellSession, invocation: CommandInvocation) -> CommandResult:
    parser = argparse.ArgumentParser(prog="audit-diff", add_help=False)
    parser.add_argument("left_event_id")
    parser.add_argument("right_event_id")
    parser.add_argument("--context", type=int, default=5)
    parser.add_argument("--json", action="store_true", dest="emit_json")
    try:
        args = parser.parse_args(invocation.args)
    except SystemExit:
        return _usage_error("Usage: audit-diff <left-event-id> <right-event-id> [--context N] [--json]")

    ledger_path = shell.root / "cli" / "data" / "snapshot_ledger.jsonl"
    diff_tool = LedgerAuditDiff(ledger_path)

    try:
        diff_result = diff_tool.diff(args.left_event_id, args.right_event_id, context=args.context)
    except KeyError as exc:
        return CommandResult(status=1, stderr=str(exc) + "\n")

    audit_payload = {
        "left": diff_result["left"],
        "right": diff_result["right"],
        "timestamp_delta_ms": diff_result["timestamp_delta_ms"],
        "payload_diff": diff_result["payload_diff"],
        "context": diff_result["context"],
    }

    if args.emit_json:
        body = json.dumps(audit_payload, ensure_ascii=False, indent=2) + "\n"
        return CommandResult(stdout=body, audit=audit_payload)

    left = diff_result["left"]
    right = diff_result["right"]
    delta_ms = diff_result["timestamp_delta_ms"]
    payload_diff = diff_result["payload_diff"]

    lines = [
        f"Ledger diff between {left['event_id']} (#{left['index']}) and {right['event_id']} (#{right['index']})",
        f"Timestamp delta: {delta_ms} ms",
    ]

    changed = payload_diff.get("changed") or {}
    only_left = payload_diff.get("only_left") or {}
    only_right = payload_diff.get("only_right") or {}

    if changed:
        lines.append("Changed fields:")
        for key, values in changed.items():
            lines.append(f"  - {key}: {values['left']!r} -> {values['right']!r}")
    if only_left:
        lines.append("Only in left:")
        for key, value in only_left.items():
            lines.append(f"  - {key}: {value!r}")
    if only_right:
        lines.append("Only in right:")
        for key, value in only_right.items():
            lines.append(f"  - {key}: {value!r}")

    context_left = diff_result["context"].get("left") or []
    context_right = diff_result["context"].get("right") or []
    if context_left:
        lines.append("Context preceding left:")
        for event in context_left:
            lines.append(
                f"  - #{event['index']:>4} {event['timestamp']} {event.get('kind')} {event['event_id']}"
            )
    if context_right:
        lines.append("Context preceding right:")
        for event in context_right:
            lines.append(
                f"  - #{event['index']:>4} {event['timestamp']} {event.get('kind')} {event['event_id']}"
            )

    lines.append("")
    return CommandResult(stdout="\n".join(lines), audit=audit_payload)


@command(
    name="ledger-inspect",
    summary="Inspect snapshot ledger events",
    usage="ledger-inspect [--limit N] [--kind KIND] [--json]",
    capabilities=("process",),
)
def ledger_inspect_command(shell: ShellSession, invocation: CommandInvocation) -> CommandResult:
    parser = argparse.ArgumentParser(prog="ledger-inspect", add_help=False)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--kind", type=str, default=None)
    parser.add_argument("--json", action="store_true", dest="emit_json")
    try:
        args = parser.parse_args(invocation.args)
    except SystemExit:
        return _usage_error("Usage: ledger-inspect [--limit N] [--kind KIND] [--json]")

    ledger_path = shell.root / "cli" / "data" / "snapshot_ledger.jsonl"
    inspector = LedgerInspector(ledger_path)
    summary = inspector.summary()
    tail = inspector.tail(limit=args.limit, kind=args.kind)
    payload = {
        "summary": summary,
        "latest_events": tail,
    }

    if args.emit_json:
        body = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
        return CommandResult(stdout=body, audit=payload)

    lines = [
        f"Ledger path: {ledger_path}",
        f"Total events: {summary['total_events']}",
    ]
    if summary["by_kind"]:
        lines.append("Events by kind:")
        for kind, count in sorted(summary["by_kind"].items()):
            lines.append(f"  - {kind}: {count}")
    else:
        lines.append("No ledger events recorded yet.")

    if tail:
        lines.append("Latest events:")
        for entry in tail:
            timestamp = entry.get("timestamp") or "unknown"
            lines.append(
                f"  - #{entry['index']}: [{timestamp}] {entry['kind']} event_id={entry['event_id']}"
            )

    lines.append("")
    return CommandResult(stdout="\n".join(lines), audit=payload)


@command(
    name="os2-dev",
    summary="Developer utilities for deterministic model execution",
    usage="os2-dev <run|replay|list> [...]",
    capabilities=("process",),
)
def os2_dev_command(shell: ShellSession, invocation: CommandInvocation) -> CommandResult:
    parser = argparse.ArgumentParser(prog="os2-dev", add_help=False)
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", add_help=False)
    run_parser.add_argument("model")
    run_parser.add_argument("--prompt", dest="prompt_text")
    run_parser.add_argument("--prompt-file", dest="prompt_file")
    run_parser.add_argument("--json", action="store_true", dest="emit_json")

    replay_parser = subparsers.add_parser("replay", add_help=False)
    replay_parser.add_argument("event_id")
    replay_parser.add_argument("--json", action="store_true", dest="emit_json")

    list_parser = subparsers.add_parser("list", add_help=False)
    list_parser.add_argument("--limit", type=int, default=10)
    list_parser.add_argument("--json", action="store_true", dest="emit_json")

    try:
        args = parser.parse_args(invocation.args)
    except SystemExit:
        return _usage_error("Usage: os2-dev <run|replay|list> [...]")

    if args.command == "run":
        prompt: Optional[str] = None
        if args.prompt_text:
            prompt = args.prompt_text
        elif args.prompt_file:
            path = (shell.cwd / args.prompt_file).resolve()
            if not str(path).startswith(str(shell.root)):
                return CommandResult(status=1, stderr="Permission denied\n")
            try:
                prompt = path.read_text(encoding="utf-8")
            except OSError as exc:
                return CommandResult(status=1, stderr=f"Failed to read prompt file: {exc}\n")
        else:
            prompt = ""

        try:
            result = shell.ai_executor.execute(args.model, prompt)
        except ModelRegistryError as exc:
            return CommandResult(status=1, stderr=str(exc) + "\n")
        except Exception as exc:  # pragma: no cover - delegated
            return CommandResult(status=1, stderr=f"execution failed: {exc}\n")

        record = {
            "model": result.record.name,
            "capability": result.record.capability,
            "registry_event_id": result.registry_event_id,
            "prompt_hash": result.prompt_hash,
            "response_digest": result.inference.digest,
            "tokens": result.inference.token_count,
            "latency_ms": result.inference.latency_ms,
            "feedback": result.feedback,
        }

        if args.emit_json:
            payload = {
                **record,
                "audit": result.audit_payload(),
            }
            body = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
            return CommandResult(stdout=body, audit=payload)

        lines = [
            f"Model: {result.record.name} ({result.record.capability})",
            f"Event ID: {result.registry_event_id}",
            f"Prompt hash: {result.prompt_hash}",
            f"Response digest: {result.inference.digest}",
            f"Tokens: {result.inference.token_count}",
            f"Latency: {result.inference.latency_ms} ms",
        ]
        if result.feedback:
            lines.append("Feedback:")
            for name, feedback in result.feedback.items():
                lines.append(f"  - {name}: {feedback}")
        else:
            lines.append("Feedback: none registered")
        lines.append("")
        return CommandResult(stdout="\n".join(lines), audit=record)

    if args.command == "replay":
        try:
            inference = shell.ai_replay.replay(args.event_id)
        except AIReplayError as exc:
            return CommandResult(status=1, stderr=str(exc) + "\n")

        payload = {
            "event_id": args.event_id,
            "digest": inference.digest,
            "tokens": inference.token_count,
            "latency_ms": inference.latency_ms,
        }

        if args.emit_json:
            body = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
            return CommandResult(stdout=body, audit=payload)

        lines = [
            f"Replayed event {args.event_id}",
            f"Digest: {inference.digest}",
            f"Tokens: {inference.token_count}",
            f"Latency: {inference.latency_ms} ms",
        ]
        lines.append("")
        return CommandResult(stdout="\n".join(lines), audit=payload)

    if args.command == "list":
        records = shell.ai_replay.list_records()
        if args.limit and args.limit > 0:
            records = records[-args.limit :]
        payload = [record.payload for record in records]

        if args.emit_json:
            body = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
            return CommandResult(stdout=body, audit={"records": payload})

        if not records:
            return CommandResult(stdout="No replay records available.\n", audit={"records": []})

        lines = ["Replay records:"]
        for record in records:
            lines.append(
                f"  - {record.event_id} model={record.model} prompt_hash={record.prompt_hash} stored_at={record.stored_at}"
            )
        lines.append("")
        return CommandResult(stdout="\n".join(lines), audit={"records": payload})

    return _usage_error("Usage: os2-dev <run|replay|list> [...]")


@command(
    name="time-travel",
    summary="Inspect stored snapshot state and diffs",
    usage="time-travel [--list | --from ID --to ID] [--json]",
    capabilities=("process",),
)
def time_travel_command(shell: ShellSession, invocation: CommandInvocation) -> CommandResult:
    parser = argparse.ArgumentParser(prog="time-travel", add_help=False)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--from", dest="from_snapshot", type=int)
    parser.add_argument("--to", dest="to_snapshot", type=int)
    parser.add_argument("--json", action="store_true", dest="emit_json")
    try:
        args = parser.parse_args(invocation.args)
    except SystemExit:
        return _usage_error("Usage: time-travel [--list | --from ID --to ID] [--json]")

    debugger = TimeTravelDebugger(shell.root)

    if args.list:
        snapshots = debugger.available_snapshots()
        payload = {"snapshots": snapshots}
        if args.emit_json:
            body = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
            return CommandResult(stdout=body, audit=payload)
        if not snapshots:
            return CommandResult(stdout="No snapshots captured yet.\n", audit=payload)
        lines = ["Available snapshots:"]
        lines.extend(f"  - {snapshot}" for snapshot in snapshots)
        lines.append("")
        return CommandResult(stdout="\n".join(lines), audit=payload)

    if args.from_snapshot is None or args.to_snapshot is None:
        return _usage_error("Usage: time-travel [--list | --from ID --to ID] [--json]")

    try:
        diff = debugger.diff(args.from_snapshot, args.to_snapshot)
    except FileNotFoundError as exc:
        return CommandResult(status=1, stderr=f"Snapshot state missing: {exc}\n")

    if args.emit_json:
        body = json.dumps(diff, ensure_ascii=False, indent=2) + "\n"
        return CommandResult(stdout=body, audit=diff)

    lines = [
        f"Snapshot diff from {diff['from']} to {diff['to']}",
    ]
    if diff["added"]:
        lines.append("Added keys:")
        for key, value in diff["added"].items():
            lines.append(f"  + {key} = {value!r}")
    if diff["removed"]:
        lines.append("Removed keys:")
        for key, value in diff["removed"].items():
            lines.append(f"  - {key} = {value!r}")
    if diff["changed"]:
        lines.append("Changed keys:")
        for key, change in diff["changed"].items():
            lines.append(f"  * {key}: {change['from']!r} -> {change['to']!r}")
    if not (diff["added"] or diff["removed"] or diff["changed"]):
        lines.append("No state differences detected.")
    lines.append("")
    return CommandResult(stdout="\n".join(lines), audit=diff)


@command(
    name="python-verify",
    summary="Verify deterministic replay for Python VM sessions",
    usage="python-verify [--json] [--limit N]",
    capabilities=("admin",),
)
def python_verify_command(shell: ShellSession, invocation: CommandInvocation) -> CommandResult:
    parser = argparse.ArgumentParser(prog="python-verify", add_help=False)
    parser.add_argument("--json", action="store_true", dest="as_json")
    parser.add_argument("--limit", type=int, default=None)
    try:
        args = parser.parse_args(invocation.args)
    except SystemExit:
        return _usage_error("Usage: python-verify [--json] [--limit N]")

    try:
        result = shell.python_verifier.verify(limit=args.limit)
    except PythonDeterminismError as exc:
        return CommandResult(status=1, stderr=str(exc) + "\n")

    audit = {
        "checked_sessions": result["checked_sessions"],
        "verified_sessions": result["verified_sessions"],
        "failed_sessions": result["failed_sessions"],
        "aggregate_stdout_hash": result["aggregate_stdout_hash"],
        "aggregate_stderr_hash": result["aggregate_stderr_hash"],
    }

    if args.as_json:
        body = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
        return CommandResult(stdout=body, audit=audit)

    lines = [
        "Python VM deterministic verification",
        f"  sessions verified: {result['verified_sessions']}/{result['checked_sessions']}",
        f"  stdout hash: {result['aggregate_stdout_hash']}",
        f"  stderr hash: {result['aggregate_stderr_hash']}",
    ]
    failures = result.get("failures", [])
    if failures:
        lines.append("  failures detected:")
        for entry in failures[:5]:
            session = entry.get("session_id", "unknown")
            errors = ", ".join(entry.get("errors", []))
            lines.append(f"    - {session}: {errors}")
        if len(failures) > 5:
            lines.append(f"    (+{len(failures) - 5} more)")
    else:
        lines.append("  no failures detected")
    return CommandResult(stdout="\n".join(lines) + "\n", audit=audit)


@command(
    name="load-modules",
    summary="Load command modules",
    usage="load-modules",
)
def load_modules(shell: ShellSession, _: CommandInvocation) -> CommandResult:
    modules_dir = shell.root / "cli" / "modules"
    if not modules_dir.exists():
        return CommandResult(stdout="No modules directory\n")
    loaded: List[str] = []
    for path in modules_dir.glob("*.json"):
        spec = json.loads(path.read_text(encoding="utf-8"))
        name = spec.get("name", path.stem)
        signature = spec.get("signature")
        token_id = spec.get("token_id", "default")
        algorithm = spec.get("signature_algorithm", "hmac-sha256")
        if not signature:
            return CommandResult(status=1, stderr=f"Module {name} missing signature\n")

        canonical = dict(spec)
        canonical.pop("signature", None)
        canonical.pop("token_id", None)
        canonical.pop("signature_algorithm", None)
        digest_source = json.dumps(canonical, ensure_ascii=False, sort_keys=True)
        digest = hashlib.sha256(digest_source.encode("utf-8")).hexdigest()
        try:
            shell.signature_verifier.verify_digest(
                digest,
                signature,
                key_id=token_id,
                algorithm=algorithm,
            )
        except SignatureError as exc:
            return CommandResult(status=1, stderr=f"Signature verification failed for {name}: {exc}\n")

        for cmd in spec.get("commands", []):
            cmd_name = cmd["name"]
            summary = cmd.get("summary", f"Module command {cmd_name}")
            usage = cmd.get("usage", cmd_name)
            required_caps = cmd.get("capabilities", ["basic"])
            response = cmd.get("response", "")

            def module_handler(shell: ShellSession, invocation: CommandInvocation, response=response) -> CommandResult:
                return CommandResult(stdout=response.format(args=" ".join(invocation.args)) + "\n")

            shell.register(
                Command(
                    name=cmd_name,
                    summary=LocalizedString(summary),
                    usage={"default": usage},
                    handler=module_handler,
                    required_capabilities=required_caps,
                )
            )
        loaded.append(name)
    if not loaded:
        return CommandResult(stdout="No modules loaded\n")
    return CommandResult(stdout="Loaded modules: " + ", ".join(loaded) + "\n")


@command(
    name="module-prune",
    summary="Detect and remove unnecessary command modules",
    usage="module-prune [--json] [--dry-run]",
    capabilities=("admin",),
)
def module_prune_command(shell: ShellSession, invocation: CommandInvocation) -> CommandResult:
    output_json = False
    dry_run = False
    for arg in invocation.args:
        if arg == "--json":
            output_json = True
        elif arg == "--dry-run":
            dry_run = True
        else:
            return _usage_error("Usage: module-prune [--json] [--dry-run]")
    if dry_run:
        report = shell.module_cleaner.analyze()
        if output_json:
            return CommandResult(
                stdout=json.dumps(report, indent=2, ensure_ascii=False) + "\n"
            )
        return CommandResult(stdout=_format_module_entropy(report, dry_run=True) + "\n")
    try:
        result = shell.module_cleaner.prune()
    except ModuleCleanerError as exc:
        return CommandResult(status=1, stderr=str(exc) + "\n")
    if output_json:
        return CommandResult(stdout=json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    return CommandResult(stdout=_format_module_entropy(result, dry_run=False) + "\n")


@command(
    name="module-perms",
    summary="Manage Python module permissions for capability tokens",
    usage="module-perms [--json] <list|grant|revoke> ...",
    capabilities=("admin",),
)
def module_permissions_command(shell: ShellSession, invocation: CommandInvocation) -> CommandResult:
    args = list(invocation.args)
    output_json = False
    if args and args[0] == "--json":
        output_json = True
        args = args[1:]
    if not args:
        return _usage_error("Usage: module-perms [--json] <list|grant|revoke> ...")

    action = args[0]
    registry = shell.module_permissions

    if action == "list":
        permissions = registry.list_permissions()
        audit = {"permissions": permissions}
        if output_json:
            body = json.dumps({"permissions": permissions}, indent=2, ensure_ascii=False)
            return CommandResult(stdout=body + "\n", audit=audit)
        lines = ["Module permissions:"]
        for token_id in sorted(permissions):
            entries = ", ".join(sorted(permissions[token_id])) or "(none)"
            lines.append(f"  {token_id}: {entries}")
        return CommandResult(stdout="\n".join(lines) + "\n", audit=audit)

    if action == "grant":
        if len(args) != 3:
            return _usage_error("Usage: module-perms [--json] grant <token_id> <module>")
        token_id, module_name = args[1:3]
        try:
            result = registry.grant(token_id, module_name)
        except ModulePermissionError as exc:
            return CommandResult(status=1, stderr=str(exc) + "\n")
        audit = {
            "action": "grant",
            "token_id": result["token_id"],
            "module": result["module"],
            "ledger_event_id": (result.get("ledger_event") or {}).get("event_id")
            if result.get("ledger_event")
            else None,
        }
        if output_json:
            payload = {k: v for k, v in result.items() if k != "ledger_event"}
            return CommandResult(stdout=json.dumps(payload, indent=2, ensure_ascii=False) + "\n", audit=audit)
        message = f"Granted module {result['module']} to token {result['token_id']}"
        return CommandResult(stdout=message + "\n", audit=audit)

    if action == "revoke":
        if len(args) != 3:
            return _usage_error("Usage: module-perms [--json] revoke <token_id> <module>")
        token_id, module_name = args[1:3]
        try:
            result = registry.revoke(token_id, module_name)
        except ModulePermissionError as exc:
            return CommandResult(status=1, stderr=str(exc) + "\n")
        audit = {
            "action": "revoke",
            "token_id": result["token_id"],
            "module": result["module"],
            "ledger_event_id": (result.get("ledger_event") or {}).get("event_id")
            if result.get("ledger_event")
            else None,
        }
        if output_json:
            payload = {k: v for k, v in result.items() if k != "ledger_event"}
            return CommandResult(stdout=json.dumps(payload, indent=2, ensure_ascii=False) + "\n", audit=audit)
        message = f"Revoked module {result['module']} from token {result['token_id']}"
        return CommandResult(stdout=message + "\n", audit=audit)

    return _usage_error("Usage: module-perms [--json] <list|grant|revoke> ...")


@command(
    name="hash-ledger",
    summary="Inspect or toggle the snapshot ledger read-only guard",
    usage="hash-ledger [--json] <status|lock|unlock>",
    capabilities=("admin",),
)
def hash_ledger_command(shell: ShellSession, invocation: CommandInvocation) -> CommandResult:
    args = list(invocation.args)
    output_json = False
    if args and args[0] == "--json":
        output_json = True
        args = args[1:]
    if not args:
        return _usage_error("Usage: hash-ledger [--json] <status|lock|unlock>")

    action = args[0]
    ledger = shell.snapshot_ledger

    if action == "status":
        state = "read-only" if ledger.is_read_only() else "writable"
        audit = {"mode": state}
        if output_json:
            return CommandResult(stdout=json.dumps({"mode": state}, indent=2) + "\n", audit=audit)
        return CommandResult(stdout=f"Snapshot ledger is {state}\n", audit=audit)

    if action == "lock":
        if ledger.is_read_only():
            return CommandResult(stdout="Snapshot ledger already read-only\n", audit={"mode": "read-only"})
        event = ledger.record_event({"kind": "snapshot_ledger_mode_changed", "mode": "read_only"})
        ledger.set_read_only(True)
        audit = {"mode": "read-only", "ledger_event_id": event.get("event_id")}
        if output_json:
            return CommandResult(stdout=json.dumps(audit, indent=2) + "\n", audit=audit)
        return CommandResult(stdout="Snapshot ledger set to read-only\n", audit=audit)

    if action == "unlock":
        if not ledger.is_read_only():
            return CommandResult(stdout="Snapshot ledger already writable\n", audit={"mode": "writable"})
        ledger.set_read_only(False)
        event = ledger.record_event({"kind": "snapshot_ledger_mode_changed", "mode": "writable"})
        audit = {"mode": "writable", "ledger_event_id": event.get("event_id")}
        if output_json:
            return CommandResult(stdout=json.dumps(audit, indent=2) + "\n", audit=audit)
        return CommandResult(stdout="Snapshot ledger unlocked for writes\n", audit=audit)

    return _usage_error("Usage: hash-ledger [--json] <status|lock|unlock>")


@command(
    name="living-system",
    summary="Transition the kernel into the living deterministic system stage",
    usage="living-system [status|transition] [--json] [--refresh] [--notes TEXT] [--operator NAME] [--force]",
    capabilities=("admin",),
)
def living_system_command(shell: ShellSession, invocation: CommandInvocation) -> CommandResult:
    args = list(invocation.args)
    action = "status"
    if args and not args[0].startswith("--"):
        action = args.pop(0)

    output_json = False
    refresh = False
    force = False
    notes = ""
    operator = shell.user

    while args:
        arg = args.pop(0)
        if arg == "--json":
            output_json = True
        elif arg == "--refresh":
            refresh = True
        elif arg == "--force":
            force = True
        elif arg == "--notes":
            if not args:
                return _usage_error(
                    "Usage: living-system [status|transition] [--json] [--refresh] [--notes TEXT] [--operator NAME] [--force]"
                )
            notes = args.pop(0)
        elif arg == "--operator":
            if not args:
                return _usage_error(
                    "Usage: living-system [status|transition] [--json] [--refresh] [--notes TEXT] [--operator NAME] [--force]"
                )
            operator = args.pop(0)
        else:
            return _usage_error(
                "Usage: living-system [status|transition] [--json] [--refresh] [--notes TEXT] [--operator NAME] [--force]"
            )

    if action == "status":
        payload = shell.living_system.status(refresh=refresh)
        if output_json:
            return CommandResult(stdout=json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
        return CommandResult(stdout=_format_living_system_state(payload) + "\n")
    if action == "transition":
        try:
            state = shell.living_system.transition(
                operator=operator,
                notes=notes,
                force=force,
            )
        except LivingDeterministicSystemError as exc:
            return CommandResult(status=1, stderr=str(exc) + "\n")
        if output_json:
            return CommandResult(stdout=json.dumps(state, indent=2, ensure_ascii=False) + "\n")
        return CommandResult(stdout=_format_living_system_state(state) + "\n")

    return _usage_error(
        "Usage: living-system [status|transition] [--json] [--refresh] [--notes TEXT] [--operator NAME] [--force]"
    )


@command(
    name="publish-shell-manual",
    summary="Publish the deterministic shell technical guide",
    usage="publish-shell-manual [--json]",
    capabilities=("admin",),
)
def publish_shell_manual_command(
    shell: ShellSession, invocation: CommandInvocation
) -> CommandResult:
    args = list(invocation.args)
    emit_json = False
    if args and args[0] == "--json":
        emit_json = True
        args = args[1:]
    if args:
        return _usage_error("Usage: publish-shell-manual [--json]")
    try:
        result = shell.documentation.publish_shell_manual()
    except DocumentationError as exc:
        return CommandResult(status=1, stderr=str(exc) + "\n")
    doc_path = result["path"].relative_to(shell.root).as_posix()
    audit = {
        "action": "publish-shell-manual",
        "doc_path": doc_path,
        "ledger_event": result.get("event"),
    }
    if emit_json:
        payload = {
            "doc_path": doc_path,
            "ledger_event": result.get("event"),
        }
        return CommandResult(
            stdout=json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            audit=audit,
        )
    return CommandResult(
        stdout=f"Published deterministic shell manual at {doc_path}\n",
        audit=audit,
    )


@command(
    name="document-release-workflow",
    summary="Generate the deterministic release workflow reference",
    usage="document-release-workflow [--json]",
    capabilities=("admin",),
)
def document_release_workflow_command(
    shell: ShellSession, invocation: CommandInvocation
) -> CommandResult:
    args = list(invocation.args)
    emit_json = False
    if args and args[0] == "--json":
        emit_json = True
        args = args[1:]
    if args:
        return _usage_error("Usage: document-release-workflow [--json]")
    try:
        result = shell.documentation.publish_release_workflow()
    except DocumentationError as exc:
        return CommandResult(status=1, stderr=str(exc) + "\n")
    doc_path = result["path"].relative_to(shell.root).as_posix()
    audit = {
        "action": "document-release-workflow",
        "doc_path": doc_path,
        "ledger_event": result.get("event"),
    }
    if emit_json:
        payload = {
            "doc_path": doc_path,
            "ledger_event": result.get("event"),
        }
        return CommandResult(
            stdout=json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            audit=audit,
        )
    return CommandResult(
        stdout=f"Documented release workflow at {doc_path}\n",
        audit=audit,
    )


@command(
    name="document-module-tree",
    summary="Automate module tree documentation via Roken Assembly",
    usage="document-module-tree [--json]",
    capabilities=("admin",),
)
def document_module_tree_command(
    shell: ShellSession, invocation: CommandInvocation
) -> CommandResult:
    args = list(invocation.args)
    emit_json = False
    if args and args[0] == "--json":
        emit_json = True
        args = args[1:]
    if args:
        return _usage_error("Usage: document-module-tree [--json]")
    try:
        result = shell.documentation.document_module_tree()
    except DocumentationError as exc:
        return CommandResult(status=1, stderr=str(exc) + "\n")
    doc_path = result["path"].relative_to(shell.root).as_posix()
    modules = result.get("modules", [])
    audit = {
        "action": "document-module-tree",
        "doc_path": doc_path,
        "module_count": len(modules),
        "ledger_event": result.get("event"),
    }
    if emit_json:
        payload = {
            "doc_path": doc_path,
            "modules": modules,
            "ledger_event": result.get("event"),
        }
        return CommandResult(
            stdout=json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            audit=audit,
        )
    lines = [f"Documented {len(modules)} module manifest(s) at {doc_path}"]
    for entry in modules:
        lines.append(
            " - {name} ({command_count} commands, token={token_id})".format(**entry)
        )
    return CommandResult(stdout="\n".join(lines) + "\n", audit=audit)


@command(
    name="kernel-ready-flag",
    summary="Manage the kernel readiness flag for the next evolution",
    usage="kernel-ready-flag [--json] [--set|--clear]",
    capabilities=("admin",),
)
def kernel_ready_flag_command(
    shell: ShellSession, invocation: CommandInvocation
) -> CommandResult:
    args = list(invocation.args)
    emit_json = False
    action = "status"
    while args:
        arg = args.pop(0)
        if arg == "--json":
            emit_json = True
        elif arg == "--set":
            action = "set"
        elif arg == "--clear":
            action = "clear"
        else:
            return _usage_error("Usage: kernel-ready-flag [--json] [--set|--clear]")

    state_path = shell.root / "cli" / "data" / "kernel_state.json"
    relative_path = state_path.relative_to(shell.root).as_posix()

    if action == "status":
        if state_path.exists():
            try:
                state = json.loads(state_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                state = {"ready_for_next_evolution": False}
        else:
            state = {"ready_for_next_evolution": False}
        payload = {
            "doc_path": relative_path,
            "ready_for_next_evolution": bool(state.get("ready_for_next_evolution")),
            "updated_at": state.get("updated_at"),
        }
        if emit_json:
            return CommandResult(
                stdout=json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                audit={"action": "kernel-ready-flag-status", **payload},
            )
        readiness = "ready" if payload["ready_for_next_evolution"] else "not ready"
        message = f"Kernel readiness flag is {readiness} (stored at {relative_path})"
        return CommandResult(
            stdout=message + "\n",
            audit={"action": "kernel-ready-flag-status", **payload},
        )

    ready_value = action == "set"
    try:
        result = shell.documentation.set_ready_flag(ready=ready_value)
    except DocumentationError as exc:
        return CommandResult(status=1, stderr=str(exc) + "\n")
    payload = {
        "doc_path": relative_path,
        "ready_for_next_evolution": ready_value,
        "ledger_event": result.get("event"),
    }
    if emit_json:
        return CommandResult(
            stdout=json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            audit={"action": "kernel-ready-flag-update", **payload},
        )
    state_text = "ready" if ready_value else "not ready"
    return CommandResult(
        stdout=f"Set kernel readiness flag to {state_text} at {relative_path}\n",
        audit={"action": "kernel-ready-flag-update", **payload},
    )


def _usage_error(message: str) -> CommandResult:
    return CommandResult(status=1, stderr=message + "\n")


def _format_model_list(records: List["ModelRecord"], output: str = "table") -> str:
    if not records:
        return "No models installed"
    if output == "json":
        payload = [record.to_dict() for record in records]
        return json.dumps(payload, indent=2, ensure_ascii=False)
    name_width = max(4, max((len(record.name) for record in records), default=4))
    provider_width = max(8, max((len(record.provider) for record in records), default=8))
    capability_width = max(10, max((len(record.capability) for record in records), default=10))
    header = (
        f"{'NAME'.ljust(name_width)}  "
        f"{'PROVIDER'.ljust(provider_width)}  "
        f"{'CAPABILITY'.ljust(capability_width)}  SOURCE"
    )
    lines = [header, "-" * len(header)]
    for record in records:
        lines.append(
            "  ".join(
                [
                    record.name.ljust(name_width),
                    record.provider.ljust(provider_width),
                    record.capability.ljust(capability_width),
                    record.source,
                ]
            )
        )
    return "\n".join(lines)


def _format_module_entropy(report: Dict[str, object], *, dry_run: bool) -> str:
    counts = report.get("counts", {})
    total = int(counts.get("total", len(report.get("modules", []))))
    removable = int(counts.get("removable", len(report.get("removable", []))))
    lines = [
        f"Total modules: {total}",
        f"Removable modules: {removable}",
    ]
    if dry_run:
        candidates = report.get("removable", [])
        if not candidates:
            lines.append("No modules require pruning")
            return "\n".join(lines)
        lines.append("Candidates:")
        for entry in candidates:
            reason = entry.get("reason") or "unspecified"
            lines.append(f" - {entry.get('name')} ({reason})")
        return "\n".join(lines)
    removed = report.get("removed", [])
    lines.append(f"Removed modules: {len(removed)}")
    if not removed:
        lines.append("No modules were removed")
        return "\n".join(lines)
    lines.append("Removed:")
    for entry in removed:
        reason = entry.get("reason") or "unspecified"
        path = entry.get("path") or ""
        path_suffix = f" at {path}" if path else ""
        lines.append(f" - {entry.get('name')} ({reason}){path_suffix}")
    return "\n".join(lines)


def _format_living_system_state(payload: Dict[str, Any]) -> str:
    if "state" in payload and isinstance(payload["state"], dict):
        state = payload["state"]
        observation = payload.get("observation")
    else:
        state = payload
        observation = None

    lines = []
    stage = state.get("current_stage", "dormant")
    lines.append(f"Current stage: {stage}")
    activated_at = state.get("activated_at") or "not yet activated"
    lines.append(f"Activated at: {activated_at}")
    operator = state.get("operator") or "unknown"
    lines.append(f"Operator: {operator}")
    notes = state.get("notes")
    if notes:
        lines.append(f"Notes: {notes}")
    readiness = state.get("readiness", {})
    ready_components = int(readiness.get("ready_components", 0))
    total_components = int(readiness.get("total_components", 0))
    readiness_state = readiness.get("state", "unknown")
    minimum_ready = int(readiness.get("minimum_ready", 0))
    lines.append(
        "Readiness: "
        f"{ready_components}/{total_components} ready (state: {readiness_state}, minimum {minimum_ready})"
    )
    pending = readiness.get("pending_components") or []
    if pending:
        lines.append("Pending components: " + ", ".join(sorted(str(name) for name in pending)))
    components = state.get("components", {})
    if components:
        lines.append("Component states:")
        for name in sorted(components):
            details = components[name]
            comp_state = details.get("state", "unknown")
            lines.append(f"  - {name.replace('_', ' ')}: {comp_state}")
    ledger_event = state.get("ledger_event")
    if isinstance(ledger_event, dict) and ledger_event.get("event_id"):
        lines.append(f"Ledger event: {ledger_event['event_id']}")
    if observation and isinstance(observation, dict):
        observed_at = observation.get("observed_at", "unknown")
        obs_readiness = observation.get("readiness", {}).get("state", "unknown")
        lines.append(f"Observation ({observed_at}): {obs_readiness}")
    return "\n".join(lines)


def _parse_key_value_flags(args: List[str], prefix: str) -> Tuple[Dict[str, Any], List[str]]:
    metadata: Dict[str, Any] = {}
    remaining: List[str] = []
    iterator = iter(args)
    for token in iterator:
        if token.startswith(prefix):
            key = token[len(prefix) :]
            try:
                value = next(iterator)
            except StopIteration as exc:  # pragma: no cover - defensive
                raise ValueError(f"Missing value for {token}") from exc
            metadata[key] = value
        else:
            remaining.append(token)
    return metadata, remaining


_TASK_CHECKBOX_RE = re.compile(r"^- \[(?P<status>.)\]")


def _summarize_task_list(body: str) -> Dict[str, int]:
    completed = 0
    in_progress = 0
    not_started = 0

    for raw_line in body.splitlines():
        line = raw_line.lstrip()
        match = _TASK_CHECKBOX_RE.match(line)
        if not match:
            continue
        status = match.group("status")
        if status.lower() == "x":
            completed += 1
        elif status in {"~", ":"}:
            in_progress += 1
        else:
            not_started += 1

    total = completed + in_progress + not_started
    remaining = total - completed
    return {
        "total": total,
        "completed": completed,
        "in_progress": in_progress,
        "not_started": not_started,
        "remaining": remaining,
    }


@command(
    name="task-progress",
    summary="Display roadmap task completion totals",
    usage="task-progress [--json]",
)
def task_progress(shell: ShellSession, invocation: CommandInvocation) -> CommandResult:
    if invocation.args and invocation.args != ["--json"]:
        return _usage_error("Usage: task-progress [--json]")

    task_file = shell.root / "docs" / "yol_hikayesi.md"
    try:
        body = task_file.read_text(encoding="utf-8")
    except FileNotFoundError:
        return CommandResult(status=1, stderr="yol_hikayesi.md not found\n")

    summary = _summarize_task_list(body)
    if summary["total"] == 0:
        return CommandResult(stdout="No tasks found in docs/yol_hikayesi.md\n")

    audit = {
        "command": "task-progress",
        "task_progress": summary,
    }

    if invocation.args == ["--json"]:
        payload = json.dumps(summary, indent=2, sort_keys=True)
        return CommandResult(stdout=payload + "\n", audit=audit)

    lines = [
        f"Total tasks: {summary['total']}",
        f"Completed: {summary['completed']}",
    ]
    if summary["in_progress"]:
        lines.append(f"In progress: {summary['in_progress']}")
    if summary["not_started"]:
        lines.append(f"Not started: {summary['not_started']}")
    lines.append(f"Remaining: {summary['remaining']}")

    return CommandResult(stdout="\n".join(lines) + "\n", audit=audit)


@command(
    name="kernel-task-rates",
    summary="Measure task completion rates from kernel logs",
    usage="kernel-task-rates [--json]",
)
def kernel_task_rates(shell: ShellSession, invocation: CommandInvocation) -> CommandResult:
    if invocation.args and invocation.args != ["--json"]:
        return _usage_error("Usage: kernel-task-rates [--json]")

    analyzer = KernelTaskAnalyzer(shell.kernel_log.path)
    summary = analyzer.compute()
    payload = summary.to_dict()
    audit = {
        "command": "kernel-task-rates",
        "task_rates": payload,
    }

    if invocation.args == ["--json"]:
        body = json.dumps(payload, indent=2, sort_keys=True)
        return CommandResult(stdout=body + "\n", audit=audit)

    lines = [
        f"Total tasks: {summary.total}",
        f"Attempted: {summary.attempted}",
        f"Success: {summary.success}",
        f"Failure: {summary.failure}",
    ]
    if summary.skipped:
        lines.append(f"Skipped: {summary.skipped}")
    rate_percent = summary.success_rate * 100
    lines.append(f"Success rate: {rate_percent:.2f}%")

    return CommandResult(stdout="\n".join(lines) + "\n", audit=audit)


@command(
    name="os2",
    summary="OS2 model management and inference",
    usage="os2 <model|prompt> ...",
)
def os2_command(shell: ShellSession, invocation: CommandInvocation) -> CommandResult:
    if not invocation.args:
        return _usage_error("Usage: os2 <model|prompt> ...")
    category, *rest = invocation.args
    if category == "model":
        return _handle_os2_model(shell, rest)
    if category == "prompt":
        return _handle_os2_prompt(shell, rest)
    return _usage_error(f"Unknown os2 category: {category}")


def _handle_os2_model(shell: ShellSession, args: List[str]) -> CommandResult:
    if not args:
        return _usage_error("Usage: os2 model <install|list|remove> ...")
    action, *rest = args
    if action == "install":
        return _handle_model_install(shell, rest)
    if action == "list":
        return _handle_model_list(shell, rest)
    if action == "remove":
        return _handle_model_remove(shell, rest)
    return _usage_error(f"Unknown model action: {action}")


def _handle_model_install(shell: ShellSession, args: List[str]) -> CommandResult:
    if not args:
        return _usage_error(
            "Usage: os2 model install <name> [--source SRC] [--provider PROVIDER] [--manifest PATH] "
            "[--capability CAP] [--force] [--fetch-metadata] [--meta-<key> <value>]"
        )
    name = args[0]
    options = args[1:]
    try:
        metadata, remainder = _parse_key_value_flags(options, "--meta-")
    except ValueError as exc:
        return CommandResult(status=1, stderr=str(exc) + "\n")
    source = "huggingface"
    provider = "huggingface"
    manifest_path: Optional[Path] = None
    overwrite = False
    fetch_hf_metadata = False
    capability: Optional[str] = None
    idx = 0
    while idx < len(remainder):
        token = remainder[idx]
        if token == "--source":
            idx += 1
            if idx >= len(remainder):
                return _usage_error("--source requires a value")
            source = remainder[idx]
        elif token == "--provider":
            idx += 1
            if idx >= len(remainder):
                return _usage_error("--provider requires a value")
            provider = remainder[idx]
        elif token == "--manifest":
            idx += 1
            if idx >= len(remainder):
                return _usage_error("--manifest requires a value")
            manifest_path = Path(remainder[idx])
        elif token == "--capability":
            idx += 1
            if idx >= len(remainder):
                return _usage_error("--capability requires a value")
            capability = remainder[idx]
        elif token == "--force":
            overwrite = True
        elif token == "--fetch-metadata":
            fetch_hf_metadata = True
        else:
            return _usage_error(f"Unknown option: {token}")
        idx += 1
    manifest_record = None
    download_report = None
    manifest: Optional[str]
    if manifest_path is not None:
        resolved = manifest_path if manifest_path.is_absolute() else (shell.cwd / manifest_path)
        resolved = resolved.resolve()
        try:
            manifest_record = load_manifest(resolved, default_name=name, default_provider=provider)
        except ModelSourceError as exc:
            return CommandResult(status=1, stderr=f"Manifest error: {exc}\n")
        source = manifest_record.url
        provider = manifest_record.provider
        manifest = str(resolved)
        metadata.setdefault("source_manifest", manifest_record.to_metadata())
        metadata.setdefault("token_cost", manifest_record.token_cost)
        if manifest_record.capability and not capability:
            capability = manifest_record.capability
        models_dir = shell.root / "cli" / "models" / name
        try:
            download_report = shell.download_manager.download(name, manifest_record, models_dir)
        except TokenBudgetExceeded as exc:
            return CommandResult(status=1, stderr=f"Token budget exceeded: {exc}\n")
        except DownloadError as exc:
            return CommandResult(status=1, stderr=f"Download failed: {exc}\n")
        metadata.setdefault("download", download_report.to_metadata())
    else:
        manifest = None

    if fetch_hf_metadata and provider.lower() == "huggingface":
        try:
            hf_metadata = shell.hf_client.fetch_model_metadata(name)
        except HuggingFaceAPIError as exc:
            shell.logger.warning("Failed to fetch Hugging Face metadata: %s", exc)
            metadata.setdefault("huggingface_error", str(exc))
        else:
            metadata.setdefault("huggingface", hf_metadata)

    try:
        runtime_plan = shell.runtime_loader.plan(name)
    except RuntimeLoaderError as exc:
        return CommandResult(status=1, stderr=f"Runtime detection failed: {exc}\n")
    metadata.setdefault("runtime", runtime_plan.to_metadata())

    try:
        record = shell.model_registry.install(
            name,
            source=source,
            provider=provider,
            manifest=manifest,
            metadata=metadata,
            overwrite=overwrite,
            capability=capability,
        )
    except ModelRegistryError as exc:
        return CommandResult(status=1, stderr=str(exc) + "\n")
    shell.capabilities.add(record.capability)
    audit = {
        "event": "model-installed",
        "model": record.name,
        "provider": record.provider,
        "source": record.source,
        "capability": record.capability,
    }
    if download_report is not None:
        audit.update(
            {
                "download_path": str(download_report.path),
                "download_sha256": download_report.sha256,
                "download_tokens": download_report.tokens,
                "download_event_id": download_report.ledger_event.get("event_id"),
                "signature_verified": download_report.signature_verified,
            }
        )
        if download_report.snapshot_event is not None:
            audit["snapshot_event_id"] = download_report.snapshot_event.get("event_id")
        if download_report.cas_path is not None:
            audit["cas_path"] = str(download_report.cas_path)
        if download_report.entropy_event is not None:
            audit["entropy_event_id"] = download_report.entropy_event.get("event_id")
            audit["entropy_bits"] = download_report.entropy_event.get("entropy_bits")
    if runtime_plan is not None:
        audit.update(
            {
                "runtime_backend": runtime_plan.backend,
                "runtime_device": runtime_plan.device,
            }
        )
    stdout_lines = [
        f"Installed model {record.name}",
        f"Capability: {record.capability}",
    ]
    if download_report is not None:
        stdout_lines.append(f"Artifact stored at {download_report.path}")
        stdout_lines.append(f"Tokens consumed: {download_report.tokens}")
        if download_report.cas_path is not None:
            stdout_lines.append(f"CAS entry: {download_report.cas_path}")
        stdout_lines.append(
            "Signature verified" if download_report.signature_verified else "Signature not provided"
        )
        if download_report.entropy_event is not None:
            entropy_bits = download_report.entropy_event.get("entropy_bits")
            stdout_lines.append(f"Entropy logged: {entropy_bits} bits")
    stdout_lines.append(f"Runtime backend: {runtime_plan.backend} ({runtime_plan.device})")
    return CommandResult(
        stdout="\n".join(stdout_lines) + "\n",
        audit=audit,
    )


def _handle_model_list(shell: ShellSession, args: List[str]) -> CommandResult:
    output_format = "table"
    if args:
        if len(args) == 1 and args[0] == "--json":
            output_format = "json"
        else:
            return _usage_error("Usage: os2 model list [--json]")
    records = shell.model_registry.list()
    body = _format_model_list(records, output=output_format)
    return CommandResult(stdout=body + "\n")


def _handle_model_remove(shell: ShellSession, args: List[str]) -> CommandResult:
    if not args:
        return _usage_error("Usage: os2 model remove <name>")
    name = args[0]
    try:
        record = shell.model_registry.remove(name)
    except ModelRegistryError as exc:
        return CommandResult(status=1, stderr=str(exc) + "\n")
    shell.capabilities.discard(record.capability)
    audit = {"event": "model-removed", "model": name, "capability": record.capability}
    return CommandResult(
        stdout=(
            f"Removed model {name}\n"
            f"Capability released: {record.capability}\n"
        ),
        audit=audit,
    )


def _handle_os2_prompt(shell: ShellSession, args: List[str]) -> CommandResult:
    if len(args) < 2:
        return _usage_error("Usage: os2 prompt <model> <text>")
    model_name = args[0]
    prompt_text = " ".join(args[1:])
    try:
        result = shell.ai_executor.execute(model_name, prompt_text)
    except ModelRegistryError as exc:
        return CommandResult(status=1, stderr=str(exc) + "\n")
    except ModelArtifactVerificationError as exc:
        return CommandResult(status=1, stderr=f"Artifact verification failed: {exc}\n")
    audit = result.audit_payload()
    return CommandResult(stdout=result.completion + "\n", audit=audit)


# -------------------- background execution ------------------


def execute_background(shell: ShellSession, line: str) -> Job:
    def target() -> None:
        result = execute_pipeline(shell, line)
        job.result = result
        job.status = "completed" if result.status == 0 else "failed"

    thread = threading.Thread(target=target, daemon=True)
    job = shell.add_job(line, thread)
    thread.start()
    return job


# -------------------- REPL loop ------------------------------


class Completer:
    def __init__(self, shell: ShellSession) -> None:
        self.shell = shell

    def complete(self, text: str, state: int) -> Optional[str]:
        buffer = readline.get_line_buffer()
        tokens = shlex.split(buffer, posix=True)
        if buffer.endswith(" "):
            tokens.append("")
        if not tokens or len(tokens) == 1:
            options = [name for name in self.shell.registry.names() if name.startswith(text)]
        else:
            base = Path(text)
            directory = base.parent if text else Path(".")
            options = []
            try:
                for entry in (self.shell.cwd / directory).iterdir():
                    name = entry.name
                    if not text or name.startswith(base.name):
                        suffix = "/" if entry.is_dir() else ""
                        options.append(str(directory / (name + suffix)))
            except FileNotFoundError:
                options = []
        if state < len(options):
            return options[state]
        return None


class Shell:
    def __init__(self, root: Path) -> None:
        self.session = ShellSession(root)
        self._register_commands()
        self.completer = Completer(self.session)
        readline.set_completer(self.completer.complete)
        readline.parse_and_bind("tab: complete")
        history_path = root / ".os2_shell_history"
        try:
            readline.read_history_file(history_path)
        except FileNotFoundError:
            pass
        self.history_path = history_path

    def _register_commands(self) -> None:
        for obj in globals().values():
            if callable(obj) and hasattr(obj, "__command_definition__"):
                self.session.register(obj.__command_definition__)

    def prompt(self) -> str:
        try:
            rel = self.session.cwd.relative_to(self.session.root)
        except ValueError:
            rel = self.session.cwd
        rel_text = "." if str(rel) == "." else str(rel)
        return f"os2:{rel_text}$ "

    def run(self) -> None:
        self.session.stream_python_output = True
        try:
            while True:
                try:
                    line = input(self.prompt())
                except EOFError:
                    print()
                    break
                except KeyboardInterrupt:
                    print()
                    continue
                line = line.strip()
                if not line:
                    continue
                background = False
                if line.endswith("&"):
                    line = line[:-1].strip()
                    background = True
                if background:
                    job = execute_background(self.session, line)
                    print(f"[{job.job_id}] {job.thread.ident} {job.command_line}")
                    continue
                result = execute_pipeline(self.session, line)
                if result.stdout and not result.streamed:
                    print(result.stdout, end="")
                if result.stderr and not result.streamed:
                    print(result.stderr, end="", file=sys.stderr)
        finally:
            self.session.stream_python_output = False
            readline.write_history_file(self.history_path)
            self.session.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    args_list = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(prog="cli.command_shell", add_help=True)
    parser.add_argument("--script", dest="script", metavar="PATH", help="Run commands from a script file relative to the repository root")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="Command to execute non-interactively")
    parsed = parser.parse_args(args_list)

    root = Path(__file__).resolve().parents[1]
    shell = Shell(root)

    if parsed.script:
        invocation = CommandInvocation(name="run-script", args=[parsed.script])
        result = run_script(shell.session, invocation)
        if result.stdout and not result.streamed:
            print(result.stdout, end="")
        if result.stderr and not result.streamed:
            print(result.stderr, end="", file=sys.stderr)
        shell.session.close()
        return result.status

    if parsed.command:
        line = " ".join(parsed.command)
        result = execute_pipeline(shell.session, line)
        if result.stdout and not result.streamed:
            print(result.stdout, end="")
        if result.stderr and not result.streamed:
            print(result.stderr, end="", file=sys.stderr)
        shell.session.close()
        return result.status

    shell.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
