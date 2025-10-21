"""Runtime loader that selects an execution backend for installed models."""

from __future__ import annotations

import os
import platform
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class HardwareProfile:
    """Description of the detected hardware relevant for model execution."""

    backend: str
    devices: List[str]
    reason: str

    def to_metadata(self) -> Dict[str, str | List[str]]:
        return {"backend": self.backend, "devices": list(self.devices), "reason": self.reason}


@dataclass
class RuntimePlan:
    """Represents the runtime decision for a model installation."""

    model: str
    backend: str
    device: str
    hardware: HardwareProfile

    def to_metadata(self) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "model": self.model,
            "backend": self.backend,
            "device": self.device,
        }
        payload["hardware"] = self.hardware.to_metadata()
        return payload


class RuntimeLoaderError(RuntimeError):
    """Raised when the runtime loader cannot determine an execution plan."""


class RuntimeLoader:
    """Detect available hardware and provide a runtime plan for models."""

    def __init__(self, *, env: Optional[Dict[str, str]] = None) -> None:
        self._env = env or os.environ

    # Public API -----------------------------------------------------
    def detect_profile(self) -> HardwareProfile:
        """Inspect the current machine and determine the preferred backend."""

        devices: List[str] = []
        reason = ""

        # Attempt to use PyTorch CUDA availability if torch is installed.
        try:
            import torch  # type: ignore

            if torch.cuda.is_available():  # pragma: no cover - optional dependency
                device_count = torch.cuda.device_count()
                devices = [torch.cuda.get_device_name(i) for i in range(device_count)]
                reason = "CUDA available via torch"
                return HardwareProfile(backend="cuda", devices=devices, reason=reason)
        except ModuleNotFoundError:
            reason = "torch not installed"
        except Exception as exc:  # pragma: no cover - defensive
            reason = f"torch detection failed: {exc}"[:200]

        # Fallback to environment heuristics for GPU detection.
        cuda_env = self._env.get("CUDA_VISIBLE_DEVICES")
        rocm_env = self._env.get("ROCR_VISIBLE_DEVICES")
        if cuda_env and cuda_env.strip("- "):
            devices = [entry.strip() for entry in cuda_env.split(",") if entry.strip()]
            return HardwareProfile(backend="cuda", devices=devices, reason="CUDA_VISIBLE_DEVICES set")
        if rocm_env and rocm_env.strip("- "):
            devices = [entry.strip() for entry in rocm_env.split(",") if entry.strip()]
            return HardwareProfile(backend="rocm", devices=devices, reason="ROCR_VISIBLE_DEVICES set")

        # Apple Silicon detection via platform tags.
        if platform.system() == "Darwin" and platform.processor().lower().startswith("arm"):
            devices = ["mps:0"]
            return HardwareProfile(backend="mps", devices=devices, reason="Apple Silicon detected")

        cpu_brand = platform.processor() or platform.machine()
        if not cpu_brand:
            cpu_brand = "generic-cpu"
        devices = [cpu_brand]
        return HardwareProfile(backend="cpu", devices=devices, reason=reason or "default cpu fallback")

    def plan(self, model: str, *, preferred_device: Optional[str] = None) -> RuntimePlan:
        """Create a runtime plan for the provided model name."""

        profile = self.detect_profile()
        device = preferred_device or (profile.devices[0] if profile.devices else "cpu")
        backend = profile.backend
        if backend == "cuda" and device == "cpu":
            raise RuntimeLoaderError("CUDA backend selected without GPU device")
        return RuntimePlan(model=model, backend=backend, device=device, hardware=profile)


__all__ = [
    "HardwareProfile",
    "RuntimeLoader",
    "RuntimeLoaderError",
    "RuntimePlan",
]
