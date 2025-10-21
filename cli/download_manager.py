"""Secure download manager with token budget enforcement."""

from __future__ import annotations

import hashlib
import json
import threading
import time
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

from .cas import ContentAddressableStore, ContentAddressableStoreError
from .model_sources import ModelSource
from .signature import SignatureError, SignatureVerifier
from .snapshot_ledger import SnapshotLedger


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _isoformat_utc(dt: datetime) -> str:
    return dt.isoformat().replace("+00:00", "Z")


class DownloadError(RuntimeError):
    """Raised when the download manager cannot fetch or validate an artifact."""


class TokenBudgetExceeded(RuntimeError):
    """Raised when the token budget for downloads would be exceeded."""


@dataclass
class DownloadReport:
    """Summary of a successful download operation."""

    path: Path
    sha256: str
    size_bytes: int
    tokens: int
    ledger_event: Dict[str, object]
    snapshot_event: Optional[Dict[str, object]] = None
    cas_path: Optional[Path] = None
    signature_verified: bool = False
    entropy_event: Optional[Dict[str, object]] = None

    def to_metadata(self) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "path": str(self.path),
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "tokens": self.tokens,
            "ledger_event": dict(self.ledger_event),
            "signature_verified": self.signature_verified,
        }
        if self.snapshot_event is not None:
            payload["snapshot_event"] = dict(self.snapshot_event)
        if self.cas_path is not None:
            payload["cas_path"] = str(self.cas_path)
        if self.entropy_event is not None:
            payload["entropy_event"] = dict(self.entropy_event)
        return payload


class TokenBudgetLedger:
    """Append-only ledger tracking token consumption for downloads."""

    def __init__(
        self,
        path: Path,
        *,
        limit: int = 200_000,
        period_seconds: int = 24 * 60 * 60,
    ) -> None:
        self._path = path
        self._limit = limit
        self._period = period_seconds
        self._lock = threading.RLock()
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def _iter_entries(self) -> list[Dict[str, object]]:
        if not self._path.exists():
            return []
        entries = []
        for line in self._path.read_text(encoding="utf-8").splitlines():
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(entry, dict):
                entries.append(entry)
        return entries

    def _recent_entries(self, now: Optional[float] = None) -> list[Dict[str, object]]:
        now_ts = now or time.time()
        recent: list[Dict[str, object]] = []
        for entry in self._iter_entries():
            ts = entry.get("ts")
            try:
                dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
            except ValueError:
                continue
            if now_ts - dt.timestamp() <= self._period:
                recent.append(entry)
        return recent

    def ensure_can_consume(self, tokens: int) -> None:
        with self._lock:
            recent = self._recent_entries()
            spent = sum(int(entry.get("tokens", 0)) for entry in recent)
            if spent + tokens > self._limit:
                raise TokenBudgetExceeded(
                    f"Token budget exceeded ({spent + tokens} > {self._limit})"
                )

    def record(self, tokens: int, context: Dict[str, object]) -> Dict[str, object]:
        event = dict(context)
        event["tokens"] = int(tokens)
        event["ts"] = _isoformat_utc(_now_utc())
        payload = json.dumps(event, sort_keys=True, ensure_ascii=False)
        event_id = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        event["event_id"] = event_id
        with self._lock:
            with self._path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, ensure_ascii=False) + "\n")
        return event


class SecureDownloadManager:
    """Download artifacts over HTTPS with SHA256 validation and token ledgering."""

    def __init__(
        self,
        ledger: TokenBudgetLedger,
        *,
        chunk_size: int = 65536,
        snapshot_ledger: Optional[SnapshotLedger] = None,
        cas: Optional[ContentAddressableStore] = None,
        signature_verifier: Optional[SignatureVerifier] = None,
    ) -> None:
        self._ledger = ledger
        self._chunk_size = chunk_size
        self._snapshot_ledger = snapshot_ledger
        self._cas = cas
        self._signature_verifier = signature_verifier

    def download(self, model: str, source: ModelSource, destination_dir: Path) -> DownloadReport:
        if not source.url.startswith("https://"):
            raise DownloadError("Only HTTPS downloads are permitted")

        self._ledger.ensure_can_consume(source.token_cost)

        destination_dir.mkdir(parents=True, exist_ok=True)
        filename = destination_dir / source.artifact
        temp_path = filename.with_suffix(filename.suffix + ".part")

        sha256 = hashlib.sha256()
        size = 0
        try:
            with urllib.request.urlopen(source.url) as response:  # nosec: B310 - validated scheme
                with temp_path.open("wb") as handle:
                    while True:
                        chunk = response.read(self._chunk_size)
                        if not chunk:
                            break
                        handle.write(chunk)
                        sha256.update(chunk)
                        size += len(chunk)
        except Exception as exc:  # pragma: no cover - network errors
            if temp_path.exists():
                temp_path.unlink(missing_ok=True)
            raise DownloadError(f"Failed to download {source.url}: {exc}") from exc

        digest = sha256.hexdigest()
        if digest.lower() != source.sha256.lower():
            temp_path.unlink(missing_ok=True)
            raise DownloadError(
                "SHA256 mismatch: "
                f"expected {source.sha256.lower()} got {digest.lower()}"
            )

        temp_path.replace(filename)

        signature_verified = False
        if self._signature_verifier and source.signature:
            try:
                self._signature_verifier.verify_digest(
                    digest,
                    source.signature,
                    key_id=source.signature_key,
                    algorithm=source.signature_algorithm,
                )
                signature_verified = True
            except SignatureError as exc:
                filename.unlink(missing_ok=True)
                raise DownloadError(f"Signature verification failed: {exc}") from exc

        cas_path: Optional[Path] = None
        if self._cas:
            try:
                cas_path = self._cas.store(filename, digest)
            except ContentAddressableStoreError as exc:
                filename.unlink(missing_ok=True)
                raise DownloadError(f"CAS storage failed: {exc}") from exc

        ledger_event = self._ledger.record(
            source.token_cost,
            {
                "event": "model-download",
                "model": model,
                "url": source.url,
                "artifact": str(filename),
                "sha256": digest,
                "size_bytes": size,
            },
        )
        snapshot_event = None
        entropy_event = None
        if self._snapshot_ledger:
            entropy_bits = int(size) * 8
            event_payload = {
                "kind": "download_entropy_captured",
                "event": "model-download",
                "model": model,
                "provider": source.provider,
                "source_url": source.url,
                "artifact": str(filename),
                "sha256": digest,
                "size_bytes": size,
                "entropy_bits": entropy_bits,
                "token_event_id": ledger_event.get("event_id"),
                "cas_path": str(cas_path) if cas_path else None,
                "signature_verified": signature_verified,
            }
            snapshot_event = self._snapshot_ledger.record_event(event_payload)
            entropy_event = snapshot_event

        return DownloadReport(
            path=filename,
            sha256=digest,
            size_bytes=size,
            tokens=source.token_cost,
            ledger_event=ledger_event,
            snapshot_event=snapshot_event,
            cas_path=cas_path,
            signature_verified=signature_verified,
            entropy_event=entropy_event,
        )


__all__ = [
    "DownloadError",
    "DownloadReport",
    "SecureDownloadManager",
    "TokenBudgetExceeded",
    "TokenBudgetLedger",
]
