from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from cli.ai_executor import DeterministicAIExecutor
from cli.ai_replay import AIReplayManager
from cli.gpu_manager import GPUAccessManager
from cli.kernel_log import KernelLogWriter
from cli.llm_adapter import DeterministicLLMAdapter
from cli.model_registry import ModelRegistry
from cli.runtime_loader import HardwareProfile, RuntimePlan
from cli.snapshot_ledger import SnapshotLedger


class RecordingAdapter(DeterministicLLMAdapter):
    def __init__(self) -> None:
        super().__init__()
        self.configure_called = False

    def generate(self, record, prompt):  # type: ignore[override]
        assert self.configure_called, "configure_deterministic_mode should run before inference"
        return super().generate(record, prompt)


class StubRuntimeLoader:
    def plan(self, model: str, *, preferred_device=None) -> RuntimePlan:  # type: ignore[override]
        hardware = HardwareProfile(backend="cuda", devices=["cuda:0"], reason="test")
        return RuntimePlan(model=model, backend="cuda", device="cuda:0", hardware=hardware)


def _create_registry(root: Path) -> ModelRegistry:
    data_dir = root / "cli" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    registry = ModelRegistry(data_dir / "models.json")

    artifact_dir = root / "cli" / "models" / "demo-model"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = artifact_dir / "model.bin"
    payload = b"deterministic"
    artifact_path.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()

    registry.install(
        "demo-model",
        source="local",
        provider="test-suite",
        manifest=None,
        metadata={"download": {"path": str(artifact_path), "sha256": digest}},
    )
    return registry


def test_ai_executor_logs_inference(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path
    registry = _create_registry(root)

    logs_dir = root / "rust" / "os2-kernel" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    kernel_log = KernelLogWriter.for_workspace(root)
    snapshot_ledger = SnapshotLedger(root / "cli" / "data" / "snapshot_ledger.jsonl")

    adapter = RecordingAdapter()
    stub_metadata = {
        "seed": 1,
        "seed_digest": "a" * 64,
        "random_seed": 1,
        "environment": {},
        "frameworks": {},
    }

    calls: list[str] = []

    def configure_stub(seed_material: str):
        adapter.configure_called = True
        calls.append(seed_material)
        return stub_metadata

    monkeypatch.setattr("cli.ai_executor.configure_deterministic_mode", configure_stub)

    runtime_loader = StubRuntimeLoader()
    gpu_manager = GPUAccessManager(root, snapshot_ledger)
    replay_manager = AIReplayManager(root, snapshot_ledger, registry, adapter)
    executor = DeterministicAIExecutor(
        root,
        registry,
        adapter,
        kernel_log,
        snapshot_ledger,
        runtime_loader,  # type: ignore[arg-type]
        replay_manager=replay_manager,
        gpu_manager=gpu_manager,
    )

    prompt = "hello deterministic world"
    result = executor.execute("demo-model", prompt)

    record = registry.get("demo-model")
    assert record is not None
    assert adapter.configure_called is True
    assert calls[0] == adapter.seed_material(record, prompt)
    assert result.completion.endswith(prompt)
    assert result.kernel_event["label"] == "model_inference"
    assert result.kernel_event["token_id"] == KernelLogWriter.token_id_for_capability(record.capability)
    assert result.gpu_event is not None
    assert result.gpu_event["label"] == "gpu_call"
    assert result.gpu_event["detail"]["lease_id"] == result.gpu_access.get("lease_id")
    assert result.snapshot_event["kind"] == "model_inference_snapshot"
    assert result.deterministic_metadata is stub_metadata
    assert result.context_switch_enter["label"] == "context_switch"
    assert result.context_switch_exit["detail"]["success"] is True
    assert result.hybrid_log["context_switch_enter"] == result.context_switch_enter["chain_hash"]
    assert result.hybrid_log["context_switch_exit"] == result.context_switch_exit["chain_hash"]
    assert result.gpu_access is not None
    assert result.gpu_access["lease_id"] is not None

    ledger_path = root / "cli" / "data" / "inference_ledger.jsonl"
    assert ledger_path.exists()
    entries = [json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines() if line]
    assert entries[-1]["event_id"] == result.registry_event_id
    assert entries[-1]["metadata"]["deterministic_mode"] == stub_metadata

    audit = result.audit_payload()
    assert audit["runtime_backend"] == "cuda"
    assert audit["deterministic_mode"] is stub_metadata
    assert audit["context_switch_enter_hash"] == result.context_switch_enter["chain_hash"]
    assert audit["gpu_lease_id"] == result.gpu_access["lease_id"]

    records = replay_manager.list_records()
    assert records and records[0].event_id == result.registry_event_id
