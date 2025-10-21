"""Deterministic inference adapter used for offline validation."""

import hashlib
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict

from .model_registry import ModelRecord


@dataclass
class InferenceResult:
    """Container for deterministic inference output."""

    completion: str
    digest: str
    latency_ms: int
    token_count: int
    seed_material: str

    def to_metadata(self) -> Dict[str, int | str]:
        return {
            "digest": self.digest,
            "latency_ms": self.latency_ms,
            "tokens": self.token_count,
            "seed_material": self.seed_material,
        }


class DeterministicLLMAdapter:
    """Produces deterministic responses suitable for offline testing."""

    def __init__(self, *, salt: str = "os2-llm-adapter") -> None:
        self._salt = salt
        self._feedback_hooks: Dict[str, Callable[[ModelRecord, str, InferenceResult], Dict[str, Any]]] = {}

    def seed_material(self, record: ModelRecord, prompt: str) -> str:
        """Return the seed string used for deterministic inference."""

        return f"{record.name}::{record.capability}::{prompt}::{self._salt}"

    def register_feedback_hook(
        self,
        name: str,
        callback: Callable[[ModelRecord, str, InferenceResult], Dict[str, Any]],
    ) -> None:
        """Register a feedback hook used to enrich deterministic completions."""

        self._feedback_hooks[name] = callback

    def remove_feedback_hook(self, name: str) -> None:
        """Unregister a previously registered feedback hook."""

        self._feedback_hooks.pop(name, None)

    def run_feedback_hooks(
        self,
        record: ModelRecord,
        prompt: str,
        result: InferenceResult,
    ) -> Dict[str, Any]:
        """Execute registered feedback hooks and return their reports."""

        reports: Dict[str, Any] = {}
        for name, hook in list(self._feedback_hooks.items()):
            try:
                reports[name] = hook(record, prompt, result)
            except Exception as exc:  # pragma: no cover - defensive
                reports[name] = {"error": str(exc)}
        return reports

    def generate(self, record: ModelRecord, prompt: str) -> InferenceResult:
        start = time.perf_counter()
        seed = self.seed_material(record, prompt)
        digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()
        prelude = "[deterministic-response]"
        completion = (
            f"{prelude}\n"
            f"model={record.name}\n"
            f"capability={record.capability}\n"
            f"digest={digest[:32]}\n"
            f"echo={prompt}"
        )
        latency_ms = int((time.perf_counter() - start) * 1000)
        token_count = max(1, len(prompt.split()))
        return InferenceResult(
            completion=completion,
            digest=digest,
            latency_ms=latency_ms,
            token_count=token_count,
            seed_material=seed,
        )


__all__ = ["DeterministicLLMAdapter", "InferenceResult"]
