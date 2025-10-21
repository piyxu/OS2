"""Boot-time model loader that activates kernel capabilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import MutableSet

from .model_registry import ModelRegistry


@dataclass
class BootModelLoaderReport:
    """Summarises the boot-time activation process."""

    activated_capabilities: list[str]
    already_present: list[str]


class BootModelLoader:
    """Synchronise model-backed capabilities at shell startup."""

    def __init__(self, registry: ModelRegistry) -> None:
        self._registry = registry

    def load(self, capability_set: MutableSet[str]) -> BootModelLoaderReport:
        existing = list(capability_set)
        activated = self._registry.sync_capabilities(capability_set)
        already_present = [
            capability
            for capability in (record.capability for record in self._registry.list())
            if capability in existing
        ]
        return BootModelLoaderReport(activated_capabilities=activated, already_present=already_present)


__all__ = ["BootModelLoader", "BootModelLoaderReport"]
