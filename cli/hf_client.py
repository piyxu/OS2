"""Signed Hugging Face API client for retrieving model metadata."""

from __future__ import annotations

import hashlib
import hmac
import json
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Dict, Optional


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _isoformat_utc(dt: datetime) -> str:
    return dt.isoformat().replace("+00:00", "Z")


class HuggingFaceAPIError(RuntimeError):
    """Raised when the Hugging Face client fails to fetch metadata."""


class SignedHuggingFaceClient:
    """Perform signed metadata requests against the Hugging Face API."""

    def __init__(
        self,
        *,
        base_url: str = "https://huggingface.co/api",
        token: Optional[str] = None,
        signing_key: str = "os2-hf-signing",
        user_agent: str = "OS2-CLI/0.1",
        timeout: int = 10,
        opener: Optional[urllib.request.OpenerDirector] = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._token = token
        self._signing_key = signing_key
        self._user_agent = user_agent
        self._timeout = timeout
        self._opener = opener or urllib.request.build_opener()

    def _sign(self, path: str, timestamp: str) -> str:
        payload = f"{path}:{timestamp}".encode("utf-8")
        return hmac.new(self._signing_key.encode("utf-8"), payload, hashlib.sha256).hexdigest()

    def fetch_model_metadata(self, model_id: str) -> Dict[str, object]:
        """Retrieve metadata for *model_id* from the Hugging Face API."""

        path = f"/models/{model_id}"
        url = f"{self._base_url}{path}"
        timestamp = _isoformat_utc(_now_utc())
        signature = self._sign(path, timestamp)
        headers = {
            "User-Agent": self._user_agent,
            "X-OS2-Timestamp": timestamp,
            "X-OS2-Signature": signature,
            "Accept": "application/json",
        }
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"

        request = urllib.request.Request(url, headers=headers)
        try:
            with self._opener.open(request, timeout=self._timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:  # pragma: no cover - network errors
            raise HuggingFaceAPIError(f"Failed to fetch metadata: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise HuggingFaceAPIError(f"Invalid JSON response: {exc}") from exc

        if not isinstance(payload, dict):
            raise HuggingFaceAPIError("Unexpected response payload")

        metadata: Dict[str, object] = {
            "id": payload.get("modelId", model_id),
            "author": payload.get("author"),
            "sha": payload.get("sha"),
            "last_modified": payload.get("lastModified"),
            "downloads": payload.get("downloads"),
            "likes": payload.get("likes"),
        }
        card_data = payload.get("cardData")
        if isinstance(card_data, dict):
            metadata["card_data"] = card_data
        siblings = payload.get("siblings")
        if isinstance(siblings, list):
            metadata["artifacts"] = [entry.get("rfilename") for entry in siblings if isinstance(entry, dict)]
        metadata["fetched_at"] = timestamp
        metadata["signature"] = signature
        return metadata


__all__ = ["HuggingFaceAPIError", "SignedHuggingFaceClient"]
