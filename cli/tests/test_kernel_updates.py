import hashlib
import hmac
import json
from pathlib import Path

from cli.command_shell import (
    CommandInvocation,
    ShellSession,
    kernel_updates_command,
)


def _default_signature(digest: str) -> str:
    return hmac.new(
        b"os2-kernel-signing",
        msg=digest.encode("utf-8"),
        digestmod=hashlib.sha256,
    ).hexdigest()


def test_kernel_updates_distribute_records_signed_package(tmp_path: Path) -> None:
    shell = ShellSession(tmp_path)
    package_path = tmp_path / "updates" / "kernel-0.1.1.tar.gz"
    package_path.parent.mkdir(parents=True, exist_ok=True)
    package_path.write_bytes(b"deterministic update payload")
    digest = hashlib.sha256(package_path.read_bytes()).hexdigest()
    signature = _default_signature(digest)

    try:
        result = kernel_updates_command(
            shell,
            CommandInvocation(
                name="kernel-updates",
                args=[
                    "--json",
                    "distribute",
                    "0.1.1",
                    str(package_path.relative_to(tmp_path)),
                    "--sha256",
                    digest,
                    "--token-id",
                    "default",
                    "--signature",
                    signature,
                    "--notes",
                    "First deterministic kernel rollout",
                ],
            ),
        )
        listing = kernel_updates_command(
            shell,
            CommandInvocation(name="kernel-updates", args=["--json", "list"]),
        )
    finally:
        shell.close()

    payload = json.loads(result.stdout)
    package = payload["package"]
    assert package["package_id"] == 1
    assert package["version"] == "0.1.1"
    assert package["artifact"] == str(package_path.relative_to(tmp_path))
    assert package["sha256"] == digest
    assert package["token_id"] == "default"
    assert package["notes"] == "First deterministic kernel rollout"

    listing_payload = json.loads(listing.stdout)
    assert listing_payload["packages"][0]["version"] == "0.1.1"

    data_path = tmp_path / "cli" / "data" / "kernel_updates.json"
    body = json.loads(data_path.read_text(encoding="utf-8"))
    assert body[0]["package_id"] == 1

    ledger_path = tmp_path / "cli" / "data" / "snapshot_ledger.jsonl"
    lines = ledger_path.read_text(encoding="utf-8").splitlines()
    parsed = (json.loads(line) for line in reversed(lines))
    ledger_entry = next(item for item in parsed if item.get("kind") == "kernel_update_distributed")
    assert ledger_entry["package"]["version"] == "0.1.1"


def test_kernel_updates_rejects_invalid_signature(tmp_path: Path) -> None:
    shell = ShellSession(tmp_path)
    package_path = tmp_path / "kernel.tar"
    package_path.write_bytes(b"payload")
    digest = hashlib.sha256(package_path.read_bytes()).hexdigest()

    try:
        result = kernel_updates_command(
            shell,
            CommandInvocation(
                name="kernel-updates",
                args=[
                    "distribute",
                    "0.1.2",
                    str(package_path.relative_to(tmp_path)),
                    "--sha256",
                    digest,
                    "--signature",
                    "deadbeef",
                ],
            ),
        )
    finally:
        shell.close()

    assert result.status == 1
    assert "Signature verification failed" in result.stderr
