import hashlib
import json
from pathlib import Path

from cli.ai_executor import DeterministicAIExecutor
from cli.ai_replay import AIReplayManager
from cli.gpu_manager import GPUAccessManager
from cli.kernel_log import KernelLogWriter
from cli.llm_adapter import DeterministicLLMAdapter
from cli.model_registry import ModelRegistry
from cli.runtime_loader import HardwareProfile, RuntimePlan
from cli.snapshot_ledger import SnapshotLedger


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


def test_replay_and_rollback(tmp_path: Path) -> None:
    root = tmp_path
    registry = _create_registry(root)

    logs_dir = root / "rust" / "os2-kernel" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    kernel_log = KernelLogWriter.for_workspace(root)
    snapshot_path = root / "cli" / "data" / "snapshot_ledger.jsonl"
    snapshot_ledger = SnapshotLedger(snapshot_path)

    adapter = DeterministicLLMAdapter()
    gpu_manager = GPUAccessManager(root, snapshot_ledger)
    replay_manager = AIReplayManager(root, snapshot_ledger, registry, adapter)
    runtime_loader = StubRuntimeLoader()

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

    prompt = "replay deterministic prompt"
    result = executor.execute("demo-model", prompt)

    record = replay_manager.get_record(result.registry_event_id)
    assert record.prompt == prompt
    assert record.payload["hybrid_log"]["context_switch_enter"] == result.context_switch_enter["chain_hash"]

    replay = replay_manager.replay(result.registry_event_id)
    assert replay.digest == result.inference.digest

    rollback_event = replay_manager.rollback(result.registry_event_id)
    assert rollback_event["kind"] == "model_inference_rollback"
    assert rollback_event["source_event_id"] == result.registry_event_id
    assert rollback_event["snapshot_event_id"] == result.snapshot_event["event_id"]

    ledger_entries = [
        json.loads(line)
        for line in snapshot_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    kinds = [entry.get("kind") for entry in ledger_entries]
    assert "model_inference_rollback" in kinds
