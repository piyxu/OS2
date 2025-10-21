import hashlib
import io
import json
import urllib.request
from pathlib import Path

import pytest

from cli.cas import ContentAddressableStore
from cli.download_manager import SecureDownloadManager, TokenBudgetLedger
from cli.model_sources import ModelSource
from cli.snapshot_ledger import SnapshotLedger


class _DummyResponse:
    def __init__(self, payload: bytes) -> None:
        self._buffer = io.BytesIO(payload)

    def read(self, size: int) -> bytes:
        return self._buffer.read(size)

    def __enter__(self) -> "_DummyResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


@pytest.mark.parametrize("payload", [b"deterministic-bytes", b"more-bytes"])
def test_download_records_entropy_event(tmp_path: Path, monkeypatch, payload: bytes) -> None:
    digest = hashlib.sha256(payload).hexdigest()
    token_ledger = TokenBudgetLedger(tmp_path / "tokens.jsonl", limit=1000)
    snapshot_ledger = SnapshotLedger(tmp_path / "snapshot.jsonl")
    cas = ContentAddressableStore(tmp_path / "cas")
    manager = SecureDownloadManager(
        token_ledger,
        snapshot_ledger=snapshot_ledger,
        cas=cas,
    )

    source = ModelSource(
        name="demo-model",
        provider="huggingface",
        url="https://example.com/model.bin",
        sha256=digest,
        token_cost=7,
        artifact="model.bin",
        capability=None,
        metadata={},
        signature=None,
        signature_key=None,
    )

    def fake_urlopen(url: str):
        assert url == source.url
        return _DummyResponse(payload)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    report = manager.download("demo-model", source, tmp_path / "downloads")

    assert report.path.exists()
    assert report.cas_path is not None
    assert report.entropy_event is report.snapshot_event
    assert report.entropy_event is not None
    assert report.entropy_event["kind"] == "download_entropy_captured"
    assert report.entropy_event["entropy_bits"] == len(payload) * 8
    assert report.entropy_event["source_url"] == source.url
    assert report.entropy_event["token_event_id"] == report.ledger_event["event_id"]

    metadata = report.to_metadata()
    assert metadata["entropy_event"]["entropy_bits"] == len(payload) * 8

    ledger_entries = [
        json.loads(line)
        for line in (tmp_path / "snapshot.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert ledger_entries[-1]["kind"] == "download_entropy_captured"
    assert ledger_entries[-1]["sha256"] == digest
    assert ledger_entries[-1]["size_bytes"] == len(payload)
