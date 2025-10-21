"""Self task review module for managing external AI API providers."""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from .snapshot_ledger import SnapshotLedger
from .rlhf_pipeline import (
    DeterministicMetricSnapshot,
    InteractionInput,
    RLHFPipeline,
)

PROVIDERS: List[tuple[str, str]] = [
    ("codex", "OpenAI Codex"),
    ("copilot", "GitHub Copilot"),
    ("claude", "Anthropic Claude"),
    ("gemini", "Google Gemini"),
    ("huggingface", "HuggingFace Hub"),
    ("local", "Local Models"),
]

SUCCESS_STATUSES = {"success", "succeeded", "completed", "ok"}
FAILURE_STATUSES = {"failure", "failed", "error", "timeout"}


class SelfTaskReviewError(RuntimeError):
    """Raised when the self task review module encounters an error."""


@dataclass
class ProviderRecord:
    """State maintained for each external AI API provider."""

    provider_name: str
    api_key: Optional[str] = None
    active: bool = False
    last_response_hash: Optional[str] = None
    status_counts: Dict[str, int] = field(default_factory=dict)
    total_runtime_ms: float = 0.0
    last_runtime_ms: Optional[float] = None
    success_count: int = 0
    failure_count: int = 0
    skipped_count: int = 0
    last_rlhf_decision: Optional[Dict[str, object]] = None

    def to_dict(self) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "provider_name": self.provider_name,
            "api_key": self.api_key,
            "active": self.active,
            "last_response_hash": self.last_response_hash,
            "status_counts": dict(self.status_counts),
            "total_runtime_ms": self.total_runtime_ms,
            "last_runtime_ms": self.last_runtime_ms,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "skipped_count": self.skipped_count,
        }
        payload.update(self.connection_flags())
        payload["total_tasks"] = self.total_tasks
        payload["success_rate"] = self.success_rate()
        payload["last_rlhf_decision"] = dict(self.last_rlhf_decision) if isinstance(self.last_rlhf_decision, dict) else self.last_rlhf_decision
        return payload

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "ProviderRecord":
        record = cls(
            provider_name=str(payload.get("provider_name", "")),
            api_key=payload.get("api_key") or None,
            active=bool(payload.get("active", False)),
            last_response_hash=payload.get("last_response_hash") or None,
        )
        record.status_counts = dict(payload.get("status_counts", {}) or {})
        record.total_runtime_ms = float(payload.get("total_runtime_ms", 0.0) or 0.0)
        record.last_runtime_ms = payload.get("last_runtime_ms")
        if record.last_runtime_ms is not None:
            record.last_runtime_ms = float(record.last_runtime_ms)
        record.success_count = int(payload.get("success_count", 0) or 0)
        record.failure_count = int(payload.get("failure_count", 0) or 0)
        record.skipped_count = int(payload.get("skipped_count", 0) or 0)
        decision = payload.get("last_rlhf_decision")
        record.last_rlhf_decision = decision if isinstance(decision, dict) else None
        return record

    def connection_flags(self) -> Dict[str, bool]:
        connected = bool(self.active and (self.api_key or self.provider_name == "local"))
        if self.provider_name == "codex":
            return {"codex_api_connected": connected}
        return {"api_connected": connected}

    @property
    def total_tasks(self) -> int:
        return self.success_count + self.failure_count

    def success_rate(self) -> float:
        total = self.total_tasks
        if total <= 0:
            return 0.0
        return self.success_count / total

    def register_status(self, status: str, runtime_ms: Optional[float], *, active: bool) -> str:
        normalized = status.lower()
        if not active:
            normalized = "skipped"
            self.skipped_count += 1
        else:
            if runtime_ms is not None:
                self.total_runtime_ms += runtime_ms
                self.last_runtime_ms = runtime_ms
            if normalized in SUCCESS_STATUSES:
                self.success_count += 1
            elif normalized in FAILURE_STATUSES:
                self.failure_count += 1
        self.status_counts[normalized] = self.status_counts.get(normalized, 0) + 1
        return normalized


class SelfTaskReviewModule:
    """Central control layer for external AI APIs."""

    def __init__(self, root: Path, ledger: SnapshotLedger) -> None:
        self._ledger = ledger
        self._path = root / "cli" / "data" / "self_task_review.json"
        self._policy_path = root / "cli" / "data" / "rlhf_policy.json"
        self._lock = threading.RLock()
        self._providers = self._load_or_initialize()
        self._rlhf = RLHFPipeline.from_policy_file(self._policy_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def list_providers(self) -> List[Dict[str, object]]:
        with self._lock:
            return [self._providers[name].to_dict() for name, _ in PROVIDERS]

    def enable_provider(self, provider_name: str, *, api_key: Optional[str] = None) -> Dict[str, object]:
        return self._set_provider_activity(provider_name, True, api_key=api_key)

    def disable_provider(self, provider_name: str) -> Dict[str, object]:
        return self._set_provider_activity(provider_name, False)

    def set_api_key(self, provider_name: str, api_key: Optional[str]) -> Dict[str, object]:
        with self._lock:
            provider = self._get_provider(provider_name)
            provider.api_key = api_key or None
            event = self._ledger.record_event(
                {
                    "kind": "self_task_review_provider_credentials_updated",
                    "provider_name": provider.provider_name,
                    "api_key_present": bool(api_key),
                    "registry": provider.to_dict(),
                }
            )
            self._save_locked()
            return {"provider": provider.to_dict(), "ledger_event": event}

    def record_task_event(
        self,
        provider_name: str,
        task_id: str,
        status: str,
        runtime_ms: Optional[float],
        output_hash: str,
    ) -> Dict[str, object]:
        with self._lock:
            provider = self._get_provider(provider_name)
            runtime_value: Optional[float] = None
            if runtime_ms is not None:
                try:
                    runtime_value = float(runtime_ms)
                except (TypeError, ValueError):
                    runtime_value = None
            normalized = provider.register_status(status, runtime_value, active=provider.active)
            if provider.active:
                provider.last_response_hash = output_hash
            analysis = self._build_analysis(provider)
            ledger_event = self._ledger.record_event(
                {
                    "kind": "self_task_review_task_event",
                    "provider_name": provider.provider_name,
                    "task_id": task_id,
                    "status": status if provider.active else "skipped",
                    "normalized_status": normalized,
                    "runtime_ms": runtime_value,
                    "output_hash": output_hash,
                    "provider_registry": provider.to_dict(),
                    "analysis": analysis,
                }
            )

            interaction = self._build_interaction_input(
                provider, task_id, normalized, runtime_value, output_hash
            )
            decision = self._rlhf.submit_interaction(interaction)
            audit_tail = self._rlhf.audit_log()[-1] if self._rlhf.audit_log() else None
            provider.last_rlhf_decision = {
                "decision": decision.to_dict(),
                "triggered_policies": audit_tail.get("triggered_policies", []) if audit_tail else [],
            }
            rlhf_event = self._ledger.record_event(
                {
                    "kind": "self_task_review_rlhf_decision",
                    "provider_name": provider.provider_name,
                    "task_id": task_id,
                    "normalized_status": normalized,
                    "decision": decision.to_dict(),
                    "triggered_policies": provider.last_rlhf_decision["triggered_policies"],
                }
            )

            self._save_locked()
            return {
                "provider": provider.to_dict(),
                "ledger_event": ledger_event,
                "rlhf_event": rlhf_event,
            }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _load_or_initialize(self) -> Dict[str, ProviderRecord]:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        providers = {name: ProviderRecord(provider_name=name) for name, _ in PROVIDERS}
        if self._path.exists():
            try:
                payload = json.loads(self._path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                payload = []
            if isinstance(payload, list):
                for entry in payload:
                    if isinstance(entry, dict):
                        record = ProviderRecord.from_dict(entry)
                        if record.provider_name in providers:
                            providers[record.provider_name] = record
        else:
            self._write(providers)
        return providers

    def _set_provider_activity(
        self,
        provider_name: str,
        active: bool,
        *,
        api_key: Optional[str] = None,
    ) -> Dict[str, object]:
        with self._lock:
            provider = self._get_provider(provider_name)
            provider.active = active
            if api_key is not None:
                provider.api_key = api_key or None
            event = self._ledger.record_event(
                {
                    "kind": "self_task_review_provider_status",
                    "provider_name": provider.provider_name,
                    "active": provider.active,
                    "api_key_present": bool(provider.api_key),
                    "registry": provider.to_dict(),
                }
            )
            self._save_locked()
            return {"provider": provider.to_dict(), "ledger_event": event}

    def _get_provider(self, provider_name: str) -> ProviderRecord:
        key = provider_name.lower()
        provider = self._providers.get(key)
        if not provider:
            raise SelfTaskReviewError(f"Unknown provider: {provider_name}")
        return provider

    def _build_interaction_input(
        self,
        provider: ProviderRecord,
        task_id: str,
        normalized_status: str,
        runtime_ms: Optional[float],
        output_hash: str,
    ) -> InteractionInput:
        tags = [f"provider:{provider.provider_name}", f"status:{normalized_status}"]
        if normalized_status in FAILURE_STATUSES:
            tags.append("flag:failure")
        metrics = self._provider_metrics(provider)
        score = self._status_score(normalized_status)
        prompt = f"{provider.provider_name}:{task_id}"
        response = output_hash or normalized_status
        return InteractionInput(
            interaction_id=task_id,
            prompt=prompt,
            response=response,
            tags=tags,
            model_score=score,
            metrics=metrics,
        )

    def _status_score(self, normalized_status: str) -> float:
        if normalized_status in SUCCESS_STATUSES:
            return 1.0
        if normalized_status in FAILURE_STATUSES:
            return 0.0
        if normalized_status == "skipped":
            return 0.25
        return 0.5

    def _provider_metrics(self, provider: ProviderRecord) -> DeterministicMetricSnapshot:
        total = provider.success_count + provider.failure_count
        if total <= 0:
            return DeterministicMetricSnapshot(win_rate=0.0, failure_rate=0.0, average_latency=None)
        win_rate = provider.success_count / total
        failure_rate = provider.failure_count / total
        average_latency = None
        if total > 0 and provider.total_runtime_ms > 0:
            average_latency = provider.total_runtime_ms / total
        return DeterministicMetricSnapshot(
            win_rate=win_rate,
            failure_rate=failure_rate,
            average_latency=average_latency,
        )

    def _build_analysis(self, provider: ProviderRecord) -> Dict[str, object]:
        if not provider.active:
            return {
                "mode": "provider-inactive",
                "skipped_events": provider.skipped_count,
                "status_counts": dict(provider.status_counts),
            }
        total = provider.total_tasks
        average_latency = provider.total_runtime_ms / total if total else 0.0
        return {
            "mode": "provider-active",
            "success_rate": provider.success_rate(),
            "failure_count": provider.failure_count,
            "latency_ms": {
                "average": average_latency,
                "last": provider.last_runtime_ms,
                "total": provider.total_runtime_ms,
            },
            "status_counts": dict(provider.status_counts),
        }

    def _save_locked(self) -> None:
        self._write(self._providers)

    def _write(self, providers: Dict[str, ProviderRecord]) -> None:
        data = [providers[name].to_dict() for name, _ in PROVIDERS]
        self._path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


__all__ = ["SelfTaskReviewModule", "SelfTaskReviewError", "ProviderRecord"]
