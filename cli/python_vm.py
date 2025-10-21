"""Embedded Python VM launcher for the PIYXU OS2 command shell."""

from __future__ import annotations

import builtins
import contextlib
import io
import os
import runpy
import sys
import threading
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

from cli.async_queue import DeterministicAsyncQueue, DeterministicAsyncQueueReport
from cli.import_verifier import DeterministicImportVerifier
from cli.kernel_log import KernelLogWriter
from cli.snapshot_ledger import SnapshotLedger
from cli.token_sandbox import TokenSandboxBudgetExceeded, TokenSandboxManager
from cli.sys_path_registry import PythonSysPathRegistry
from cli.snapshot_registry import PythonVMSnapshotRegistry
from cli.module_permissions import ModulePermissionRegistry


class PythonVMError(RuntimeError):
    """Raised when the embedded Python VM encounters a launch failure."""


@dataclass
class PythonVMResult:
    """Structured result returned after a Python VM session finishes."""

    session_id: str
    sandbox_id: str
    snapshot_id: int
    mode: str
    stdout: str
    stderr: str
    exit_status: int
    duration_ms: float
    return_value: Optional[Any]
    token_budget: int
    tokens_consumed: int
    events: Dict[str, Mapping[str, Any]]
    resume_from_snapshot: Optional[int]
    safe_mode: bool
    snapshot_state_keys: Sequence[str]
    snapshot_state_path: Optional[str]
    module: Optional[str] = None
    streamed: bool = False


def _is_snapshot_serializable(value: Any, *, depth: int = 0) -> bool:
    if depth >= 6:
        return False
    if value is None:
        return True
    if isinstance(value, (bool, int, float, str)):
        return True
    if isinstance(value, (list, tuple)):
        if len(value) > 256:
            return False
        return all(_is_snapshot_serializable(item, depth=depth + 1) for item in value)
    if isinstance(value, dict):
        if len(value) > 256:
            return False
        for key, item in value.items():
            if not isinstance(key, str):
                return False
            if not _is_snapshot_serializable(item, depth=depth + 1):
                return False
        return True
    return False


def _extract_snapshot_state(globals_dict: Mapping[str, Any]) -> Dict[str, Any]:
    snapshot: Dict[str, Any] = {}
    for key, value in globals_dict.items():
        if key.startswith("__"):
            continue
        if _is_snapshot_serializable(value):
            snapshot[key] = value
    return snapshot


def _default_builtins() -> Dict[str, Any]:
    """Return a deterministic set of built-ins for sandbox execution."""

    allowed: Iterable[str] = (
        "abs",
        "all",
        "any",
        "bool",
        "bytes",
        "callable",
        "dict",
        "enumerate",
        "filter",
        "float",
        "format",
        "frozenset",
        "getattr",
        "hasattr",
        "hash",
        "hex",
        "int",
        "isinstance",
        "issubclass",
        "iter",
        "len",
        "list",
        "map",
        "max",
        "min",
        "next",
        "object",
        "open",
        "ord",
        "pow",
        "print",
        "range",
        "repr",
        "reversed",
        "round",
        "set",
        "slice",
        "sorted",
        "str",
        "sum",
        "tuple",
        "type",
        "zip",
        "input",
        "Exception",
        "ValueError",
        "RuntimeError",
        "NameError",
        "AssertionError",
        "BaseException",
        "__build_class__",
        "__import__",
    )
    safe: Dict[str, Any] = {}
    for name in allowed:
        if hasattr(builtins, name):
            safe[name] = getattr(builtins, name)
    return safe


class _TeeStream(io.TextIOBase):
    """Stream wrapper that mirrors writes into a buffer and an output stream."""

    def __init__(self, buffer: io.StringIO, mirror: Optional[io.TextIOBase]) -> None:
        super().__init__()
        self._buffer = buffer
        self._mirror = mirror
        self._lock = threading.RLock()

    def write(self, s: str) -> int:  # type: ignore[override]
        if not s:
            return 0
        with self._lock:
            self._buffer.write(s)
            if self._mirror is not None:
                self._mirror.write(s)
                self._mirror.flush()
        return len(s)

    def flush(self) -> None:  # type: ignore[override]
        with self._lock:
            self._buffer.flush()
            if self._mirror is not None:
                self._mirror.flush()

    def writable(self) -> bool:  # type: ignore[override]
        return True

    @property
    def encoding(self) -> str:  # type: ignore[override]
        if self._mirror is not None and hasattr(self._mirror, "encoding"):
            return self._mirror.encoding  # type: ignore[return-value]
        return "utf-8"

    def isatty(self) -> bool:  # type: ignore[override]
        if self._mirror is not None and hasattr(self._mirror, "isatty"):
            try:
                if self._mirror.isatty():  # type: ignore[misc]
                    return True
            except Exception:  # pragma: no cover - defensive
                return True
        return True

    def fileno(self) -> int:  # type: ignore[override]
        if self._mirror is not None and hasattr(self._mirror, "fileno"):
            return self._mirror.fileno()  # type: ignore[return-value]
        raise io.UnsupportedOperation("Underlying stream does not expose fileno()")


class _SysPathInjector:
    """Context manager that temporarily prepends paths to ``sys.path``."""

    def __init__(self, paths: Sequence[str]) -> None:
        self._paths: list[str] = []
        self._injected: list[str] = []
        unique: list[str] = []
        for path in paths:
            if not path:
                continue
            if path not in unique:
                unique.append(path)
        self._paths = unique

    def __enter__(self) -> "_SysPathInjector":
        injected: list[str] = []
        for path in self._paths:
            if path not in sys.path:
                sys.path.insert(0, path)
                injected.append(path)
        self._injected = injected
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        for path in self._injected:
            while path in sys.path:
                try:
                    sys.path.remove(path)
                except ValueError:  # pragma: no cover - defensive cleanup
                    break
        self._injected = []

    @property
    def injected_paths(self) -> Sequence[str]:
        return list(self._injected)


class PythonVMLauncher:
    """Launch Python code inside a deterministic sandbox."""

    def __init__(
        self,
        root: Path,
        ledger: SnapshotLedger,
        *,
        default_token_budget: int = 200,
        kernel_log: Optional[KernelLogWriter] = None,
        module_permissions: Optional[ModulePermissionRegistry] = None,
    ) -> None:
        self._root = root.resolve()
        self._ledger = ledger
        self._sessions_dir = self._root / "cli" / "python_vm"
        self._sessions_dir.mkdir(parents=True, exist_ok=True)
        self._counter_path = self._sessions_dir / "session_counter"
        self._lock = threading.RLock()
        self._builtins = _default_builtins()
        self._counter = self._load_counter()
        self._default_import = builtins.__import__
        self._sandboxes = TokenSandboxManager(
            self._sessions_dir / "sandboxes",
            ledger,
            default_budget=default_token_budget,
        )
        self._import_verifier = DeterministicImportVerifier(self._root, ledger)
        self._syspath_registry = PythonSysPathRegistry(self._root, ledger)
        self._kernel_log = kernel_log or KernelLogWriter.for_workspace(self._root)
        self._snapshot_registry = PythonVMSnapshotRegistry(self._root, ledger)
        self._module_permissions = module_permissions or ModulePermissionRegistry(self._root, ledger)

    @property
    def kernel_log_path(self) -> Path:
        return self._kernel_log.path

    # ------------------------------------------------------------------
    def _handle_system_exit(self, exc: SystemExit, stderr_buffer: io.StringIO) -> int:
        code = exc.code
        if code is None:
            return 0
        if isinstance(code, int):
            if code < 0:
                return 0
            if code > 255:
                return 255
            return code
        message: Optional[str]
        if isinstance(code, bytes):
            try:
                message = code.decode("utf-8", errors="replace")
            except Exception:  # pragma: no cover - extremely defensive
                message = str(code)
        else:
            message = str(code)
        if message:
            if not message.endswith("\n"):
                message += "\n"
            stderr_buffer.write(message)
        return 1

    # ------------------------------------------------------------------
    def _load_counter(self) -> int:
        try:
            return int(self._counter_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return 0
        except ValueError:
            return 0

    def _next_session_id(self) -> str:
        with self._lock:
            self._counter += 1
            self._counter_path.write_text(str(self._counter), encoding="utf-8")
            return f"pyvm-{self._counter:05d}"

    # ------------------------------------------------------------------
    def launch(
        self,
        *,
        expr: Optional[str] = None,
        script_path: Optional[Path] = None,
        module: Optional[str] = None,
        script_args: Optional[Sequence[str]] = None,
        globals_override: Optional[Mapping[str, Any]] = None,
        token_budget: Optional[int] = None,
        capabilities: Optional[Sequence[str]] = None,
        command_alias: Optional[str] = None,
        resume_snapshot: Optional[int] = None,
        safe_mode: bool = False,
        stream_output: bool = False,
    ) -> PythonVMResult:
        """Execute code inside the deterministic sandbox.

        When *resume_snapshot* is provided the previous snapshot state is
        hydrated before running new source. Supplying only a resume identifier
        rehydrates the copy-on-write memory tree without executing additional
        code, emitting ledger events for observability.
        """

        provided_sources = [value for value in (expr, script_path, module) if value]
        if len(provided_sources) > 1:
            raise PythonVMError("Provide only one of expr, script_path, or module")
        if not expr and not script_path and module is None and resume_snapshot is None:
            raise PythonVMError("No Python source provided for launch")
        if resume_snapshot is not None and resume_snapshot <= 0:
            raise PythonVMError("Resume snapshot must be a positive integer")

        session_id = self._next_session_id()
        args = list(script_args or [])
        resume_only = resume_snapshot is not None and expr is None and script_path is None
        if module is not None:
            resume_only = False

        if script_path is not None:
            mode = "script"
        elif module is not None:
            mode = "module"
        elif resume_only:
            mode = "resume"
        else:
            mode = "expr"

        script_label = "<pyvm>"
        if mode == "resume" and resume_snapshot is not None:
            script_label = f"<resume:{resume_snapshot}>"

        resolved_script: Optional[Path] = None
        source = ""
        if script_path is not None:
            resolved = script_path.resolve()
            if not str(resolved).startswith(str(self._root)):
                raise PythonVMError("Script path escapes workspace root")
            if not resolved.exists():
                raise PythonVMError(f"Script not found: {resolved}")
            source = resolved.read_text(encoding="utf-8")
            resolved_script = resolved
            script_label = str(resolved)
        elif expr is not None:
            source = expr
        elif module is not None:
            source = module
            script_label = f"<module:{module}>"
        else:
            source = ""

        resume_state: Optional[Dict[str, Any]] = None
        if resume_snapshot is not None:
            resume_state = self._snapshot_registry.load_state(resume_snapshot)
            if resume_state is None:
                raise PythonVMError(
                    f"Snapshot {resume_snapshot} cannot be resumed – state payload missing"
                )

        builtins_map: Dict[str, Any] = dict(self._builtins)
        if globals_override:
            override_builtins = globals_override.get("__builtins__")
            if isinstance(override_builtins, Mapping):
                builtins_map.update(override_builtins)

        globals_dict: Dict[str, Any] = {
            "__name__": "__main__",
            "__file__": script_label,
            "__builtins__": builtins_map,
        }
        if globals_override:
            globals_dict.update({k: v for k, v in globals_override.items() if k != "__builtins__"})
        argv_seed = list(args)
        if module is not None:
            argv_seed = [module, *args]
        elif resolved_script is not None:
            argv_seed = [str(resolved_script), *args]
        elif expr is not None:
            argv_seed = ["-c", *args]
        locals_dict: Dict[str, Any] = {"argv": list(argv_seed)}

        resume_event: Optional[Dict[str, Any]] = None
        if resume_snapshot is not None:
            resume_event = self._ledger.record_event(
                {
                    "kind": "python_vm_snapshot_resume_requested",
                    "session_id": session_id,
                    "resume_from_snapshot": resume_snapshot,
                    "safe_mode": bool(safe_mode),
                    "hydrate_only": resume_only,
                }
            )

        reservation = self._snapshot_registry.reserve(session_id)

        sandbox = self._sandboxes.create(
            session_id=session_id,
            token_budget=token_budget,
            capabilities=capabilities,
            snapshot_id=reservation.snapshot_id,
            resume_from_snapshot=resume_snapshot,
            safe_mode=safe_mode,
        )
        metadata_updates: Dict[str, object] = {
            "mode": mode,
            "script": script_label,
            "args": list(args),
            "snapshot_id": reservation.snapshot_id,
        }
        if module is not None:
            metadata_updates["module"] = module
        if resume_snapshot is not None:
            metadata_updates["resume_from_snapshot"] = int(resume_snapshot)
        if resume_event is not None:
            metadata_updates["resume_event_id"] = resume_event.get("event_id")
        if safe_mode:
            metadata_updates["safe_mode"] = True
            metadata_updates["network_access"] = "disabled"
        sandbox._update_metadata(metadata_updates)

        async_queue = DeterministicAsyncQueue(
            ledger=self._ledger,
            session_id=session_id,
            sandbox_id=sandbox.sandbox_id,
            snapshot_id=reservation.snapshot_id,
            kernel_log=self._kernel_log,
        )
        builtins_map["async_queue"] = async_queue
        globals_dict["async_queue"] = async_queue
        locals_dict["async_queue"] = async_queue

        permission_tokens = [
            capability.split(":", 1)[1]
            for capability in (capabilities or ())
            if capability.startswith("token:")
        ]

        builtins_map["__import__"] = self._import_verifier.create_import_hook(
            session_id=session_id,
            sandbox=sandbox,
            default_import=self._default_import,
            permission_tokens=permission_tokens,
            permission_manager=self._module_permissions,
        )

        if resume_state:
            globals_dict.update({k: v for k, v in resume_state.items() if not k.startswith("__")})
            sandbox._update_metadata({"resume_state_keys": sorted(resume_state.keys())})

        estimated_tokens = max(1, len(source.encode("utf-8")) // 4 + 1)
        try:
            sandbox.reserve(estimated_tokens)
        except TokenSandboxBudgetExceeded as exc:
            sandbox.finalize(status="budget_exceeded")
            self._snapshot_registry.cancel(session_id)
            raise PythonVMError(str(exc)) from exc

        start_event = self._ledger.record_event(
            {
                "kind": "python_vm_start",
                "session_id": session_id,
                "mode": mode,
                "script": script_label,
                "args": args,
                "sandbox_id": sandbox.sandbox_id,
                "token_budget": sandbox.token_budget,
                "tokens_reserved": sandbox.tokens_consumed,
                "snapshot_id": reservation.snapshot_id,
                "resume_from_snapshot": resume_snapshot,
                "safe_mode": bool(safe_mode),
            }
        )

        queue_created_event = async_queue.open()
        sandbox._update_metadata(
            {
                "async_queue": {
                    "queue_id": async_queue.queue_id,
                    "ledger_event_ids": {
                        "created": queue_created_event.get("event_id")
                        if queue_created_event
                        else None
                    },
                }
            }
        )

        resume_hydrated_event: Optional[Dict[str, Any]] = None
        if resume_state:
            resume_hydrated_event = self._ledger.record_event(
                {
                    "kind": "python_vm_snapshot_resume_hydrated",
                    "session_id": session_id,
                    "sandbox_id": sandbox.sandbox_id,
                    "resume_from_snapshot": resume_snapshot,
                    "hydrated_keys": sorted(resume_state.keys()),
                    "snapshot_id": reservation.snapshot_id,
                }
            )

        snapshot_event = self._snapshot_registry.record_tag(
            session_id=session_id,
            sandbox_id=sandbox.sandbox_id,
            mode=mode,
            script=script_label,
            command_alias=command_alias,
            script_args=args,
            ledger_event_ids={
                "sandbox_created": sandbox.creation_event.get("event_id")
                if sandbox.creation_event
                else None,
                "start": start_event.get("event_id"),
            },
            resume_from=resume_snapshot,
        )
        sandbox._update_metadata(
            {
                "snapshot_event_id": snapshot_event.get("event_id"),
                "snapshot_tag": snapshot_event,
            }
        )

        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        stdout_target: io.TextIOBase
        stderr_target: io.TextIOBase
        if stream_output:
            stdout_target = _TeeStream(stdout_buffer, original_stdout)
            stderr_target = _TeeStream(stderr_buffer, original_stderr)
        else:
            stdout_target = stdout_buffer
            stderr_target = stderr_buffer
        status = 0
        return_value: Optional[Any] = None
        start_ts = time.perf_counter()

        path_injections = [str(self._root)]
        if resolved_script is not None:
            path_injections.insert(0, str(resolved_script.parent))

        syspath_event: Optional[Dict[str, Any]] = None

        queue_report: Optional[DeterministicAsyncQueueReport] = None
        queue_summary: Optional[Dict[str, Any]] = None

        original_safe_env: Optional[str] = None
        if safe_mode:
            original_safe_env = os.environ.get("OS2_SAFE_MODE")
        original_sys_argv = list(sys.argv)

        try:
            if safe_mode:
                os.environ["OS2_SAFE_MODE"] = "1"
            with _SysPathInjector(path_injections) as injector, contextlib.redirect_stdout(
                stdout_target
            ), contextlib.redirect_stderr(stderr_target):
                sys.argv = list(argv_seed)
                syspath_event = self._syspath_registry.synchronize(
                    session_id=session_id,
                    sandbox_id=sandbox.sandbox_id,
                    sys_path=list(sys.path),
                    injected_paths=injector.injected_paths,
                    snapshot_id=reservation.snapshot_id,
                )
                if syspath_event:
                    sandbox._update_metadata(
                        {
                            "sys_path": [
                                entry["raw"] for entry in syspath_event.get("paths", [])
                            ],
                            "sys_path_hash": syspath_event.get("paths_hash"),
                            "sys_path_event_id": syspath_event.get("event_id"),
                        }
                    )
                merge_locals = True
                if resume_only:
                    if resume_snapshot is not None:
                        print(
                            f"[resume] snapshot {resume_snapshot} hydrated into session {session_id}",
                            file=stdout_buffer,
                        )
                else:
                    try:
                        if expr is not None:
                            compiled = compile(source, script_label, "eval")
                            return_value = eval(compiled, globals_dict, locals_dict)
                            if return_value is not None:
                                print(return_value, file=stdout_buffer)
                        elif module is not None:
                            module_globals = runpy.run_module(
                                module,
                                run_name="__main__",
                                alter_sys=True,
                            )
                            globals_dict.update(module_globals)
                        else:
                            compiled = compile(source, script_label, "exec")
                            exec(compiled, globals_dict, locals_dict)
                    except SystemExit as exc:
                        status = self._handle_system_exit(exc, stderr_buffer)
                        merge_locals = False
                    except BaseException:
                        raise
                if merge_locals:
                    globals_dict.update(
                        {k: v for k, v in locals_dict.items() if k != "__builtins__"}
                    )
        except BaseException as exc:  # pragma: no cover - handled in tests via stderr
            if isinstance(exc, SystemExit):
                status = self._handle_system_exit(exc, stderr_buffer)
            else:
                status = 1
                stderr_buffer.write(traceback.format_exc())
        finally:
            if safe_mode:
                if original_safe_env is None:
                    os.environ.pop("OS2_SAFE_MODE", None)
                else:
                    os.environ["OS2_SAFE_MODE"] = original_safe_env
            sys.argv = original_sys_argv

        queue_report = async_queue.drain()
        queue_summary = queue_report.to_dict()
        sandbox._update_metadata({"async_queue": queue_summary})
        if queue_report.status != "ok":
            status = 1
            for task in queue_report.tasks:
                if task.error:
                    stderr_buffer.write(f"[async:{task.name}] {task.error}\\n")

        duration_ms = (time.perf_counter() - start_ts) * 1000.0

        stdout_text = stdout_buffer.getvalue()
        stderr_text = stderr_buffer.getvalue()

        snapshot_state = _extract_snapshot_state(globals_dict)
        state_path = self._snapshot_registry.write_state(reservation.snapshot_id, snapshot_state)
        try:
            state_path_str = str(state_path.relative_to(self._root))
        except ValueError:
            state_path_str = str(state_path)
        sandbox._update_metadata(
            {
                "snapshot_state_path": state_path_str,
                "snapshot_state_keys": sorted(snapshot_state.keys()),
            }
        )
        state_event = self._ledger.record_event(
            {
                "kind": "python_vm_snapshot_state_saved",
                "session_id": session_id,
                "snapshot_id": reservation.snapshot_id,
                "state_keys": sorted(snapshot_state.keys()),
                "state_path": state_path_str,
                "resume_from_snapshot": resume_snapshot,
                "safe_mode": bool(safe_mode),
            }
        )

        completion_event = self._ledger.record_event(
            {
                "kind": "python_vm_complete",
                "session_id": session_id,
                "mode": mode,
                "script": script_label,
                "status": "error" if status else "ok",
                "duration_ms": round(duration_ms, 3),
                "stdout_len": len(stdout_text),
                "stderr_len": len(stderr_text),
                "sandbox_id": sandbox.sandbox_id,
                "tokens_consumed": sandbox.tokens_consumed,
                "snapshot_id": reservation.snapshot_id,
                "async_queue_status": queue_summary.get("status") if queue_summary else None,
                "async_queue_tasks_total": queue_summary.get("tasks_total") if queue_summary else 0,
                "async_queue_tasks_error": queue_summary.get("tasks_error") if queue_summary else 0,
                "resume_from_snapshot": resume_snapshot,
                "safe_mode": bool(safe_mode),
            }
        )

        release_event = sandbox.finalize("error" if status else "ok")

        ledger_ids = {
            "sandbox_created": sandbox.creation_event.get("event_id")
            if sandbox.creation_event
            else None,
            "start": start_event.get("event_id"),
            "syspath": syspath_event.get("event_id") if syspath_event else None,
            "complete": completion_event.get("event_id"),
            "sandbox_released": release_event.get("event_id"),
            "snapshot_tag": snapshot_event.get("event_id"),
            "async_queue_created": queue_summary.get("ledger_event_ids", {}).get("created")
            if queue_summary
            else None,
            "async_queue_drained": queue_summary.get("ledger_event_ids", {}).get("drained")
            if queue_summary
            else None,
            "resume_request": resume_event.get("event_id") if resume_event else None,
            "resume_hydrated": resume_hydrated_event.get("event_id")
            if resume_hydrated_event
            else None,
            "snapshot_state_saved": state_event.get("event_id") if state_event else None,
        }

        kernel_log_events: Dict[str, Mapping[str, Any]] = {}
        if self._kernel_log:
            stdout_event = self._kernel_log.record_python_stream(
                session_id=session_id,
                sandbox_id=sandbox.sandbox_id,
                stream="stdout",
                content=stdout_text,
                mode=mode,
                command_alias=command_alias,
                script=script_label,
                ledger_event_ids=ledger_ids,
                token_budget=sandbox.token_budget,
                tokens_consumed=sandbox.tokens_consumed,
                snapshot_id=reservation.snapshot_id,
            )
            stderr_event = self._kernel_log.record_python_stream(
                session_id=session_id,
                sandbox_id=sandbox.sandbox_id,
                stream="stderr",
                content=stderr_text,
                mode=mode,
                command_alias=command_alias,
                script=script_label,
                ledger_event_ids=ledger_ids,
                token_budget=sandbox.token_budget,
                tokens_consumed=sandbox.tokens_consumed,
                snapshot_id=reservation.snapshot_id,
            )
            session_event = self._kernel_log.record_python_session(
                session_id=session_id,
                sandbox_id=sandbox.sandbox_id,
                mode=mode,
                command_alias=command_alias,
                script=script_label,
                status="error" if status else "ok",
                duration_ms=duration_ms,
                ledger_event_ids=ledger_ids,
                stdout_event=stdout_event,
                stderr_event=stderr_event,
                token_budget=sandbox.token_budget,
                tokens_consumed=sandbox.tokens_consumed,
                script_args=tuple(args),
                snapshot_id=reservation.snapshot_id,
            )
            kernel_log_events = {
                "stdout": stdout_event,
                "stderr": stderr_event,
                "session": session_event,
            }
            sandbox._update_metadata(
                {
                    "kernel_log_events": {
                        name: event.get("chain_hash")
                        for name, event in kernel_log_events.items()
                        if event
                    },
                    "kernel_log_path": str(self._kernel_log.path),
                }
            )

        events = {
            "start": start_event,
            "complete": completion_event,
            "sandbox_created": sandbox.creation_event,
            "sandbox_released": release_event,
            "syspath": syspath_event,
            "snapshot_tag": snapshot_event,
            "async_queue": queue_summary,
            "kernel_log": kernel_log_events,
            "resume_request": resume_event,
            "resume_hydrated": resume_hydrated_event,
            "snapshot_state": state_event,
        }

        return PythonVMResult(
            session_id=session_id,
            sandbox_id=sandbox.sandbox_id,
            snapshot_id=reservation.snapshot_id,
            mode=mode,
            stdout=stdout_text,
            stderr=stderr_text,
            exit_status=status,
            duration_ms=duration_ms,
            return_value=return_value,
            token_budget=sandbox.token_budget,
            tokens_consumed=sandbox.tokens_consumed,
            events=events,
            resume_from_snapshot=resume_snapshot,
            safe_mode=bool(safe_mode),
            snapshot_state_keys=sorted(snapshot_state.keys()),
            snapshot_state_path=state_path_str,
            module=module,
            streamed=bool(stream_output),
        )



__all__ = ["PythonVMLauncher", "PythonVMError", "PythonVMResult"]
