from __future__ import annotations

from pathlib import Path

from cli.kernel_log import KernelLogWriter


def test_record_external_event_uses_capability_token(tmp_path: Path) -> None:
    log_path = tmp_path / "kernel.jsonl"
    state_path = tmp_path / "kernel_state.json"
    writer = KernelLogWriter(log_path, state_path=state_path)

    capability = "cap.model.demo"
    event = writer.record_external_event(
        capability=capability,
        source="unit-test",
        label="gpu_call",
        detail={"device": "cuda:0"},
    )

    assert event["kind"] == "external"
    assert event["label"] == "gpu_call"
    assert event["detail"]["source"] == "unit-test"
    assert event["detail"]["device"] == "cuda:0"
    assert event["token_id"] == KernelLogWriter.token_id_for_capability(capability)

    second = writer.record_external_event(
        capability=capability,
        source="unit-test",
        detail={},
    )
    assert second["sequence"] == event["sequence"] + 1


def test_record_context_switch(tmp_path: Path) -> None:
    log_path = tmp_path / "kernel.jsonl"
    writer = KernelLogWriter(log_path)

    enter = writer.record_context_switch(
        source="python_vm",
        target="ai_kernel",
        phase="enter",
        detail={"prompt_hash": "abc"},
        token_id=123,
    )
    exit_event = writer.record_context_switch(
        source="ai_kernel",
        target="python_vm",
        phase="exit",
        detail={"prompt_hash": "abc", "enter_chain_hash": enter["chain_hash"]},
        token_id=123,
    )

    assert enter["label"] == "context_switch"
    assert exit_event["detail"]["enter_chain_hash"] == enter["chain_hash"]
    assert exit_event["sequence"] == enter["sequence"] + 1
