"""Deterministic AI execution orchestrator used by the OS2 shell."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Dict, Optional

from .ai_determinism import configure_deterministic_mode
from .ai_replay import AIReplayManager
from .gpu_manager import GPUAccessLease, GPUAccessManager
from .kernel_log import KernelLogWriter
from .llm_adapter import DeterministicLLMAdapter, InferenceResult
from .model_registry import ModelRecord, ModelRegistry, ModelRegistryError
from .model_verifier import ArtifactVerificationResult, verify_model_artifact
from .runtime_loader import RuntimeLoader, RuntimePlan
from .snapshot_ledger import SnapshotLedger


@dataclass
class AIExecutionResult:
    """Structured response produced after executing a model prompt."""

    record: ModelRecord
    prompt: str
    completion: str
    inference: InferenceResult
    verification: ArtifactVerificationResult
    runtime_plan: RuntimePlan
    deterministic_metadata: Dict[str, Any]
    registry_event_id: str
    kernel_event: Dict[str, Any]
    snapshot_event: Dict[str, Any]
    gpu_event: Optional[Dict[str, Any]]
    prompt_hash: str
    seed_hash: str
    context_switch_enter: Dict[str, Any]
    context_switch_exit: Dict[str, Any]
    gpu_access: Optional[Dict[str, Any]]
    hybrid_log: Dict[str, Any]
    feedback: Dict[str, Any]

    def audit_payload(self) -> Dict[str, Any]:
        audit: Dict[str, Any] = {
            "event": "model-prompt",
            "model": self.record.name,
            "prompt_length": len(self.prompt),
            "capability": self.record.capability,
            "event_id": self.registry_event_id,
            "prompt_hash": self.prompt_hash,
            "artifact_verification": self.verification.to_metadata(),
            "kernel_event_sequence": self.kernel_event.get("sequence"),
            "kernel_event_chain_hash": self.kernel_event.get("chain_hash"),
            "snapshot_event_id": self.snapshot_event.get("event_id"),
            "seed_hash": self.seed_hash,
            "runtime_backend": self.runtime_plan.backend,
            "runtime_device": self.runtime_plan.device,
            "deterministic_mode": self.deterministic_metadata,
        }
        if self.gpu_event:
            audit["gpu_event_sequence"] = self.gpu_event.get("sequence")
            audit["gpu_event_chain_hash"] = self.gpu_event.get("chain_hash")
        audit["context_switch_enter_hash"] = self.context_switch_enter.get("chain_hash")
        audit["context_switch_exit_hash"] = self.context_switch_exit.get("chain_hash")
        if self.gpu_access:
            audit["gpu_lease_id"] = self.gpu_access.get("lease_id")
        if self.feedback:
            audit["feedback_providers"] = list(self.feedback.keys())
        return audit


class DeterministicAIExecutor:
    """Coordinate deterministic execution, logging, and auditing for models."""

    def __init__(
        self,
        root: Any,
        model_registry: ModelRegistry,
        llm_adapter: DeterministicLLMAdapter,
        kernel_log: KernelLogWriter,
        snapshot_ledger: SnapshotLedger,
        runtime_loader: RuntimeLoader,
        *,
        replay_manager: Optional[AIReplayManager] = None,
        gpu_manager: Optional[GPUAccessManager] = None,
    ) -> None:
        self._root = root
        self._model_registry = model_registry
        self._llm_adapter = llm_adapter
        self._kernel_log = kernel_log
        self._snapshot_ledger = snapshot_ledger
        self._runtime_loader = runtime_loader
        self._replay_manager = replay_manager
        self._gpu_manager = gpu_manager

    def execute(
        self,
        model_name: str,
        prompt: str,
        *,
        preferred_device: Optional[str] = None,
    ) -> AIExecutionResult:
        record = self._model_registry.get(model_name)
        if not record:
            raise ModelRegistryError(f"Model '{model_name}' is not installed")

        verification = verify_model_artifact(record, self._root)
        runtime_plan = self._runtime_loader.plan(record.name, preferred_device=preferred_device)

        seed_material = self._llm_adapter.seed_material(record, prompt)
        deterministic_info = configure_deterministic_mode(seed_material)
        seed_hash = hashlib.sha256(seed_material.encode("utf-8")).hexdigest()
        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()

        token_id = KernelLogWriter.token_id_for_capability(record.capability)
        context_enter = self._kernel_log.record_context_switch(
            source="python_vm",
            target="ai_kernel",
            phase="enter",
            detail={
                "model": record.name,
                "capability": record.capability,
                "prompt_hash": prompt_hash,
            },
            token_id=token_id,
        )
        context_switch_exit = context_enter

        gpu_lease: Optional[GPUAccessLease] = None
        gpu_access_detail: Optional[Dict[str, Any]] = None
        inference: Optional[InferenceResult] = None
        feedback: Dict[str, Any] = {}
        error: Optional[BaseException] = None
        try:
            if (
                runtime_plan.backend in {"cuda", "rocm", "mps"}
                and self._gpu_manager is not None
            ):
                gpu_lease = self._gpu_manager.acquire(
                    capability=record.capability,
                    backend=runtime_plan.backend,
                    device=runtime_plan.device,
                    model=record.name,
                )
            inference = self._llm_adapter.generate(record, prompt)
            feedback = self._llm_adapter.run_feedback_hooks(record, prompt, inference)
        except BaseException as exc:  # pragma: no cover - propagated to caller
            error = exc
            raise
        finally:
            context_exit = self._kernel_log.record_context_switch(
                source="ai_kernel",
                target="python_vm",
                phase="exit",
                detail={
                    "model": record.name,
                    "capability": record.capability,
                    "prompt_hash": prompt_hash,
                    "success": error is None,
                    "enter_chain_hash": context_enter.get("chain_hash"),
                },
                token_id=token_id,
            )
            if self._gpu_manager and gpu_lease is not None:
                release_event = self._gpu_manager.release(
                    gpu_lease.lease_id,
                    status="completed" if error is None else "failed",
                    tokens=inference.token_count if inference else 0,
                    detail={
                        "model": record.name,
                        "prompt_hash": prompt_hash,
                        "success": error is None,
                    },
                )
                gpu_access_detail = {
                    "lease_id": gpu_lease.lease_id,
                    "granted": gpu_lease.granted_event,
                    "released": release_event,
                }
            context_switch_exit = context_exit

        assert inference is not None  # for type checking

        metadata = inference.to_metadata()
        metadata["artifact_verification"] = verification.to_metadata()
        metadata["feedback"] = feedback
        metadata["runtime"] = runtime_plan.to_metadata()
        metadata["deterministic_mode"] = deterministic_info

        registry_event_id = self._model_registry.record_inference(
            record.name,
            prompt,
            inference.completion,
            metadata=metadata,
        )

        kernel_event = self._kernel_log.record_event(
            kind="custom",
            label="model_inference",
            detail={
                "model": record.name,
                "capability": record.capability,
                "status": "completed",
                "prompt_hash": prompt_hash,
                "response_digest": inference.digest,
                "artifact_sha256": verification.actual_sha256,
                "tokens": inference.token_count,
                "latency_ms": inference.latency_ms,
                "runtime_backend": runtime_plan.backend,
                "runtime_device": runtime_plan.device,
                "seed_hash": seed_hash,
                "feedback_providers": list(feedback.keys()),
            },
            token_id=token_id,
        )

        gpu_event: Optional[Dict[str, Any]] = None
        if runtime_plan.backend in {"cuda", "rocm", "mps"}:
            gpu_event = self._kernel_log.record_external_event(
                capability=record.capability,
                source="ai_executor",
                label="gpu_call",
                detail={
                    "model": record.name,
                    "backend": runtime_plan.backend,
                    "device": runtime_plan.device,
                    "hardware": runtime_plan.hardware.to_metadata(),
                    "tokens": inference.token_count,
                    "lease_id": gpu_access_detail.get("lease_id") if gpu_access_detail else None,
                },
            )

        snapshot_event = self._snapshot_ledger.record_event(
            {
                "kind": "model_inference_snapshot",
                "model": record.name,
                "capability": record.capability,
                "prompt_hash": prompt_hash,
                "response_digest": inference.digest,
                "seed_material": inference.seed_material,
                "seed_hash": seed_hash,
                "entropy_bits": len(inference.seed_material.encode("utf-8")) * 8,
                "artifact": verification.to_metadata(),
                "tokens": inference.token_count,
                "latency_ms": inference.latency_ms,
                "registry_event_id": registry_event_id,
                "kernel_chain_hash": kernel_event.get("chain_hash"),
                "runtime": runtime_plan.to_metadata(),
                "deterministic_mode": deterministic_info,
                "gpu_event": gpu_event.get("chain_hash") if gpu_event else None,
                "feedback": feedback,
                "hybrid_log": {
                    "context_switch_enter": context_enter.get("chain_hash"),
                    "context_switch_exit": context_switch_exit.get("chain_hash"),
                    "gpu_event_chain_hash": gpu_event.get("chain_hash") if gpu_event else None,
                    "gpu_lease_event_id": gpu_access_detail.get("lease_id") if gpu_access_detail else None,
                },
            }
        )

        hybrid_log = snapshot_event.get("hybrid_log", {})

        result = AIExecutionResult(
            record=record,
            prompt=prompt,
            completion=inference.completion,
            inference=inference,
            verification=verification,
            runtime_plan=runtime_plan,
            deterministic_metadata=deterministic_info,
            registry_event_id=registry_event_id,
            kernel_event=kernel_event,
            snapshot_event=snapshot_event,
            gpu_event=gpu_event,
            prompt_hash=prompt_hash,
            seed_hash=seed_hash,
            context_switch_enter=context_enter,
            context_switch_exit=context_switch_exit,
            gpu_access=gpu_access_detail,
            hybrid_log=hybrid_log,
            feedback=feedback,
        )

        if self._replay_manager is not None:
            self._replay_manager.record_execution(result)

        return result


__all__ = ["AIExecutionResult", "DeterministicAIExecutor"]
