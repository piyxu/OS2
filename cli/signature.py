"""Utilities for verifying signed model artifacts."""

from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass
from typing import Dict, Optional


class SignatureError(RuntimeError):
    """Raised when signature verification fails."""


def _normalise_hex(value: str) -> str:
    return value.strip().lower()


@dataclass
class SignatureKey:
    """Represents a signing key used for verification."""

    key_id: str
    secret: str


class SignatureVerifier:
    """Verify and mint signatures for deterministic artifacts."""

    def __init__(
        self,
        keys: Dict[str, str],
        *,
        default_key: Optional[str] = None,
        algorithm: str = "hmac-sha256",
    ) -> None:
        if not keys:
            raise ValueError("At least one signature key must be provided")
        self._keys = {key_id: secret for key_id, secret in keys.items()}
        self._default_key = default_key or next(iter(self._keys))
        self._algorithm = algorithm
        if algorithm != "hmac-sha256":
            raise ValueError(f"Unsupported signature algorithm: {algorithm}")

    def verify_digest(
        self,
        digest: str,
        signature: str,
        *,
        key_id: Optional[str] = None,
        algorithm: Optional[str] = None,
    ) -> None:
        """Verify *signature* for the provided artifact digest."""

        algo = algorithm or self._algorithm
        if algo != "hmac-sha256":
            raise SignatureError(f"Unsupported signature algorithm: {algo}")
        key_name = key_id or self._default_key
        if key_name not in self._keys:
            raise SignatureError(f"Unknown signature key: {key_name}")

        expected = hmac.new(
            self._keys[key_name].encode("utf-8"),
            msg=_normalise_hex(digest).encode("utf-8"),
            digestmod=hashlib.sha256,
        ).hexdigest()

        if hmac.compare_digest(expected, _normalise_hex(signature)):
            return
        raise SignatureError("Signature mismatch for artifact digest")

    def sign_digest(
        self,
        digest: str,
        *,
        key_id: Optional[str] = None,
        algorithm: Optional[str] = None,
    ) -> str:
        """Return a deterministic signature for *digest* using the stored key."""

        algo = algorithm or self._algorithm
        if algo != "hmac-sha256":
            raise SignatureError(f"Unsupported signature algorithm: {algo}")
        key_name = key_id or self._default_key
        if key_name not in self._keys:
            raise SignatureError(f"Unknown signature key: {key_name}")

        return hmac.new(
            self._keys[key_name].encode("utf-8"),
            msg=_normalise_hex(digest).encode("utf-8"),
            digestmod=hashlib.sha256,
        ).hexdigest()


__all__ = ["SignatureError", "SignatureVerifier", "SignatureKey"]
