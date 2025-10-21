"""Feedback hook registration utilities for deterministic LLM adapters."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

from .llm_adapter import DeterministicLLMAdapter, InferenceResult
from .model_registry import ModelRecord


def _load_json(path: Path) -> Dict[str, object]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass
    return {}


def register_default_feedback_hooks(adapter: DeterministicLLMAdapter, workspace: Path) -> None:
    """Register built-in feedback hooks for local evaluation loops."""

    feedback_root = workspace / "cli" / "data" / "feedback"
    llama_root = feedback_root / "llama"
    comfy_root = feedback_root / "comfyui"
    llama_root.mkdir(parents=True, exist_ok=True)
    comfy_root.mkdir(parents=True, exist_ok=True)

    def _llama_hook(record: ModelRecord, prompt: str, result: InferenceResult) -> Dict[str, object]:
        """Read deterministic feedback produced by an offline Llama adapter."""

        payload = _load_json(llama_root / f"{record.name}.json")
        if not payload:
            return {
                "available": False,
                "reason": "no-local-llama-feedback",
            }
        return {
            "available": True,
            "score": payload.get("score"),
            "notes": payload.get("notes"),
            "prompt_hash": payload.get("prompt_hash"),
            "response_digest": result.digest,
        }

    def _comfyui_hook(record: ModelRecord, prompt: str, result: InferenceResult) -> Dict[str, object]:
        """Surface the most recent ComfyUI feedback artefact if one exists."""

        payload = _load_json(comfy_root / "feedback.json")
        if not payload:
            return {
                "available": False,
                "reason": "comfyui-feedback-missing",
            }
        return {
            "available": True,
            "workflow": payload.get("workflow"),
            "rating": payload.get("rating"),
            "prompt_excerpt": prompt[:120],
            "response_digest": result.digest,
        }

    adapter.register_feedback_hook("local_llama", _llama_hook)
    adapter.register_feedback_hook("comfyui", _comfyui_hook)


__all__ = ["register_default_feedback_hooks"]
