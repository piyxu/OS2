"""Deterministic async task queue used by the embedded Python VM."""

from __future__ import annotations

import asyncio
import inspect
import json
import threading
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Mapping, Optional

from cli.kernel_log import KernelLogWriter
from cli.snapshot_ledger import SnapshotLedger


class DeterministicAsyncQueueError(RuntimeError):
    """Raised when the deterministic async queue encounters an error."""


@dataclass
class DeterministicAsyncTaskResult:
    """Result metadata for a drained asynchronous task."""

    task_id: str
    name: str
    status: str
    metadata: Dict[str, Any]
    ledger_event_ids: Dict[str, Optional[str]]
    result_repr: Optional[str] = None
    error: Optional[str] = None
    kernel_log_event: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "task_id": self.task_id,
            "name": self.name,
            "status": self.status,
            "metadata": self.metadata,
            "ledger_event_ids": {k: v for k, v in self.ledger_event_ids.items() if v},
        }
        if self.result_repr is not None:
            payload["result"] = self.result_repr
        if self.error is not None:
            payload["error"] = self.error
        if self.kernel_log_event is not None:
            payload["kernel_log_event"] = {
                "chain_hash": self.kernel_log_event.get("chain_hash"),
                "timestamp": self.kernel_log_event.get("timestamp"),
                "sequence": self.kernel_log_event.get("sequence"),
            }
        return payload


@dataclass
class DeterministicAsyncQueueReport:
    """Summary returned after draining the async queue."""

    queue_id: str
    status: str
    tasks: List[DeterministicAsyncTaskResult]
    ledger_event_ids: Dict[str, Optional[str]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "queue_id": self.queue_id,
            "status": self.status,
            "tasks_total": len(self.tasks),
            "tasks_ok": sum(1 for task in self.tasks if task.status == "ok"),
            "tasks_error": sum(1 for task in self.tasks if task.status != "ok"),
            "ledger_event_ids": {k: v for k, v in self.ledger_event_ids.items() if v},
            "tasks": [task.to_dict() for task in self.tasks],
        }


@dataclass
class _QueuedTask:
    task_id: str
    name: str
    factory: Callable[[], Awaitable[Any]]
    metadata: Dict[str, Any] = field(default_factory=dict)
    queued_event: Optional[Dict[str, Any]] = None


class DeterministicAsyncQueue:
    """Order asynchronous work deterministically within a Python VM session."""

    def __init__(
        self,
        *,
        ledger: SnapshotLedger,
        session_id: str,
        sandbox_id: str,
        snapshot_id: int,
        kernel_log: Optional[KernelLogWriter] = None,
    ) -> None:
        self._ledger = ledger
        self._session_id = session_id
        self._sandbox_id = sandbox_id
        self._snapshot_id = int(snapshot_id)
        self._kernel_log = kernel_log
        self._queue_id = f"{session_id}-async"
        self._lock = threading.RLock()
        self._counter = 0
        self._tasks: List[_QueuedTask] = []
        self._open_event: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    @property
    def queue_id(self) -> str:
        return self._queue_id

    # ------------------------------------------------------------------
    def open(self) -> Dict[str, Any]:
        """Record queue creation in the snapshot ledger."""

        if self._open_event is not None:
            return self._open_event
        event = self._ledger.record_event(
            {
                "kind": "python_vm_async_queue_created",
                "queue_id": self._queue_id,
                "session_id": self._session_id,
                "sandbox_id": self._sandbox_id,
                "snapshot_id": self._snapshot_id,
            }
        )
        self._open_event = event
        return event

    # ------------------------------------------------------------------
    def schedule(
        self,
        name: str,
        coroutine_or_factory: Callable[[], Awaitable[Any]] | Awaitable[Any],
        *,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> str:
        """Queue a coroutine for deterministic execution."""

        if not name:
            raise DeterministicAsyncQueueError("Task name is required")
        if self._open_event is None:
            raise DeterministicAsyncQueueError("Queue must be opened before scheduling tasks")

        def _to_factory(
            obj: Callable[[], Awaitable[Any]] | Awaitable[Any]
        ) -> Callable[[], Awaitable[Any]]:
            if inspect.isawaitable(obj):
                coroutine_ref = obj

                def _factory() -> Awaitable[Any]:
                    return coroutine_ref

                return _factory
            if callable(obj):
                def _factory() -> Awaitable[Any]:
                    result = obj()
                    if not inspect.isawaitable(result):
                        raise DeterministicAsyncQueueError(
                            "Scheduled callable did not return an awaitable"
                        )
                    return result

                return _factory
            raise DeterministicAsyncQueueError("Scheduled object must be awaitable or callable")

        factory = _to_factory(coroutine_or_factory)
        task_metadata = json.loads(json.dumps(dict(metadata or {}), ensure_ascii=False))

        with self._lock:
            self._counter += 1
            task_id = f"{self._session_id}-task-{self._counter:04d}"
            queued_event = self._ledger.record_event(
                {
                    "kind": "python_vm_async_task_queued",
                    "queue_id": self._queue_id,
                    "task_id": task_id,
                    "session_id": self._session_id,
                    "sandbox_id": self._sandbox_id,
                    "snapshot_id": self._snapshot_id,
                    "name": name,
                    "metadata": task_metadata,
                }
            )
            self._tasks.append(
                _QueuedTask(
                    task_id=task_id,
                    name=name,
                    factory=factory,
                    metadata=task_metadata,
                    queued_event=queued_event,
                )
            )
            return task_id

    # ------------------------------------------------------------------
    def drain(self) -> DeterministicAsyncQueueReport:
        """Execute all queued coroutines sequentially."""

        if self._open_event is None:
            raise DeterministicAsyncQueueError("Queue must be opened before draining")

        with self._lock:
            pending = list(self._tasks)
            self._tasks.clear()

        tasks_results: List[DeterministicAsyncTaskResult] = []
        overall_status = "ok"
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            for queued in pending:
                start_event = self._ledger.record_event(
                    {
                        "kind": "python_vm_async_task_started",
                        "queue_id": self._queue_id,
                        "task_id": queued.task_id,
                        "session_id": self._session_id,
                        "sandbox_id": self._sandbox_id,
                        "snapshot_id": self._snapshot_id,
                        "name": queued.name,
                    }
                )

                status = "ok"
                result_repr: Optional[str] = None
                error_text: Optional[str] = None
                kernel_log_event: Optional[Dict[str, Any]] = None

                try:
                    coroutine = queued.factory()
                    if not inspect.isawaitable(coroutine):
                        raise DeterministicAsyncQueueError(
                            "Scheduled factory did not return an awaitable"
                        )
                    result = loop.run_until_complete(coroutine)
                    result_repr = repr(result)
                except Exception as exc:  # pragma: no cover - exercised in tests
                    status = "error"
                    overall_status = "error"
                    error_text = f"{type(exc).__name__}: {exc}"
                finally:
                    complete_event = self._ledger.record_event(
                        {
                            "kind": "python_vm_async_task_completed",
                            "queue_id": self._queue_id,
                            "task_id": queued.task_id,
                            "session_id": self._session_id,
                            "sandbox_id": self._sandbox_id,
                            "snapshot_id": self._snapshot_id,
                            "name": queued.name,
                            "status": status,
                            "result_repr": result_repr,
                            "error": error_text,
                        }
                    )
                    if self._kernel_log is not None:
                        kernel_log_event = self._kernel_log.record_python_async_task(
                            session_id=self._session_id,
                            sandbox_id=self._sandbox_id,
                            queue_id=self._queue_id,
                            task_id=queued.task_id,
                            name=queued.name,
                            status=status,
                            result_repr=result_repr,
                            error=error_text,
                            ledger_event_ids={
                                "queued": queued.queued_event.get("event_id")
                                if queued.queued_event
                                else None,
                                "started": start_event.get("event_id"),
                                "completed": complete_event.get("event_id"),
                                "queue_created": self._open_event.get("event_id")
                                if self._open_event
                                else None,
                            },
                            snapshot_id=self._snapshot_id,
                        )

                tasks_results.append(
                    DeterministicAsyncTaskResult(
                        task_id=queued.task_id,
                        name=queued.name,
                        status=status,
                        metadata=queued.metadata,
                        ledger_event_ids={
                            "queued": queued.queued_event.get("event_id")
                            if queued.queued_event
                            else None,
                            "started": start_event.get("event_id"),
                            "completed": complete_event.get("event_id"),
                        },
                        result_repr=result_repr,
                        error=error_text,
                        kernel_log_event=kernel_log_event,
                    )
                )
        finally:
            try:
                loop.run_until_complete(loop.shutdown_asyncgens())
            except Exception:  # pragma: no cover - defensive cleanup
                pass
            asyncio.set_event_loop(None)
            loop.close()

        drained_event = self._ledger.record_event(
            {
                "kind": "python_vm_async_queue_drained",
                "queue_id": self._queue_id,
                "session_id": self._session_id,
                "sandbox_id": self._sandbox_id,
                "snapshot_id": self._snapshot_id,
                "tasks_total": len(tasks_results),
                "tasks_ok": sum(1 for task in tasks_results if task.status == "ok"),
                "tasks_error": sum(1 for task in tasks_results if task.status != "ok"),
                "status": overall_status,
            }
        )

        return DeterministicAsyncQueueReport(
            queue_id=self._queue_id,
            status=overall_status,
            tasks=tasks_results,
            ledger_event_ids={
                "created": self._open_event.get("event_id") if self._open_event else None,
                "drained": drained_event.get("event_id"),
            },
        )


__all__ = [
    "DeterministicAsyncQueue",
    "DeterministicAsyncQueueError",
    "DeterministicAsyncQueueReport",
    "DeterministicAsyncTaskResult",
]
