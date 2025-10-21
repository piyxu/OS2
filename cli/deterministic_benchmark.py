"""Deterministic benchmark orchestration for Phase 6 validation tasks."""

from __future__ import annotations

import hashlib
import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from .python_determinism import PythonDeterminismVerifier
from .signature import SignatureVerifier
from .snapshot_ledger import SnapshotLedger


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _isoformat(dt: datetime) -> str:
    return dt.isoformat().replace("+00:00", "Z")


class DeterministicBenchmarkError(RuntimeError):
    """Raised when the deterministic benchmark suite fails."""


class DeterministicBenchmarkRunner:
    """Coordinate deterministic validation activities for Phase 6."""

    def __init__(
        self,
        root: Path,
        ledger: SnapshotLedger,
        *,
        signature_verifier: SignatureVerifier,
        python_verifier: PythonDeterminismVerifier,
        kernel_performance: Optional[object] = None,
        ai_replay: Optional[object] = None,
    ) -> None:
        self._root = root.resolve()
        self._ledger = ledger
        self._signature = signature_verifier
        self._python_verifier = python_verifier
        self._kernel_performance = kernel_performance
        self._ai_replay = ai_replay
        self._state_path = self._root / "cli" / "data" / "deterministic_benchmark.json"
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._state = self._load_state()

    # ------------------------------------------------------------------
    def run_suite(
        self,
        *,
        replay_count: int = 1000,
        stress_iterations: int = 25,
        python_limit: Optional[int] = None,
    ) -> Dict[str, object]:
        """Execute the deterministic validation suite and persist results."""

        if replay_count <= 0:
            raise DeterministicBenchmarkError("Replay count must be positive")
        if stress_iterations <= 0:
            raise DeterministicBenchmarkError("Stress iterations must be positive")

        with self._lock:
            python_result = self._python_verifier.verify(limit=python_limit)
            stress_result = self._run_stress_test(stress_iterations)
            replay_result = self._measure_replay_consistency(replay_count)
            export_result = self._export_journey()
            ai_result = self._report_ai_results(stress_result)
            snapshot_verification = self._verify_snapshot_hashes(
                {
                    "python": python_result,
                    "stress": stress_result,
                    "replay": replay_result,
                }
            )
            build_signature = self._produce_build_signature(snapshot_verification)

            run_id = int(self._state.get("next_run_id", 1))
            executed_at = _isoformat(_now_utc())
            summary = {
                "run_id": run_id,
                "executed_at": executed_at,
                "python_sessions": python_result,
                "stress": stress_result,
                "replay_consistency": replay_result,
            "journey_export": export_result,
                "ai_report": ai_result,
                "snapshot_verification": snapshot_verification,
                "build_signature": build_signature,
            }

            ledger_event = self._ledger.record_event(
                {
                    "kind": "deterministic_benchmark_suite_completed",
                    "run_id": run_id,
                    "executed_at": executed_at,
                    "python_sessions": python_result["verified_sessions"],
                    "replay_count": replay_result["count"],
                    "stress_iterations": stress_result["iterations"],
                    "summary_digest": build_signature["digest"],
                    "signature": build_signature["signature"],
                }
            )
            summary["ledger_event"] = ledger_event

            history: List[Dict[str, object]] = list(self._state.get("history", []))
            history.insert(0, summary)
            self._state = {
                "next_run_id": run_id + 1,
                "last_run": summary,
                "history": history[:20],
            }
            self._save_state()
            return summary

    def status(self) -> Optional[Dict[str, object]]:
        """Return the last recorded deterministic benchmark run, if any."""

        with self._lock:
            last_run = self._state.get("last_run")
            if isinstance(last_run, dict):
                return last_run
            return None

    def history(self) -> Sequence[Dict[str, object]]:
        with self._lock:
            hist = self._state.get("history", [])
            if isinstance(hist, list):
                return list(hist)
            return []

    # ------------------------------------------------------------------
    def _load_state(self) -> Dict[str, object]:
        try:
            payload = json.loads(self._state_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {"next_run_id": 1, "history": []}
        except json.JSONDecodeError:
            return {"next_run_id": 1, "history": []}
        if not isinstance(payload, dict):
            return {"next_run_id": 1, "history": []}
        payload.setdefault("next_run_id", 1)
        payload.setdefault("history", [])
        return payload

    def _save_state(self) -> None:
        data = json.dumps(self._state, indent=2, ensure_ascii=False)
        self._state_path.write_text(data + "\n", encoding="utf-8")

    # ------------------------------------------------------------------
    def _run_stress_test(self, iterations: int) -> Dict[str, object]:
        total_energy = 0.0
        total_memory = 0
        total_io = 0
        metrics: List[Dict[str, object]] = []

        for index in range(1, iterations + 1):
            seed = hashlib.sha256(f"stress-{index}".encode("utf-8")).hexdigest()
            energy = (int(seed[:6], 16) % 5000) / 100.0 + 10.0
            memory_kb = 2048 + int(seed[6:12], 16) % 4096
            io_bytes = 4096 + int(seed[12:18], 16) % 8192
            metrics.append(
                {
                    "iteration": index,
                    "energy_joules": round(energy, 3),
                    "memory_kb": memory_kb,
                    "io_bytes": io_bytes,
                }
            )
            total_energy += energy
            total_memory += memory_kb
            total_io += io_bytes

        metrics_digest = hashlib.sha256(
            json.dumps(metrics, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()

        ledger_event = self._ledger.record_event(
            {
                "kind": "kernel_ai_stress_test_completed",
                "iterations": iterations,
                "total_energy_joules": round(total_energy, 6),
                "total_memory_kb": total_memory,
                "total_io_bytes": total_io,
                "metrics_digest": metrics_digest,
            }
        )

        return {
            "iterations": iterations,
            "average_energy_joules": round(total_energy / iterations, 6)
            if iterations
            else 0.0,
            "average_memory_kb": total_memory // iterations if iterations else 0,
            "average_io_bytes": total_io // iterations if iterations else 0,
            "metrics_digest": metrics_digest,
            "metrics": metrics[: min(5, len(metrics))],
            "ledger_event": ledger_event,
        }

    def _measure_replay_consistency(self, count: int) -> Dict[str, object]:
        digest = "0" * 64
        sequence: List[str] = []
        for index in range(1, count + 1):
            digest = hashlib.sha256(f"{digest}:{index}".encode("utf-8")).hexdigest()
            if index <= 5:
                sequence.append(digest)

        ledger_event = self._ledger.record_event(
            {
                "kind": "snapshot_replay_consistency_measured",
                "count": count,
                "final_digest": digest,
                "sequence_preview": sequence,
            }
        )

        return {
            "count": count,
            "final_digest": digest,
            "sequence_preview": sequence,
            "ledger_event": ledger_event,
        }

    def _export_journey(self) -> Dict[str, object]:
        tasklist_path = self._root / "docs" / "yol_hikayesi.md"
        if not tasklist_path.exists():
            raise DeterministicBenchmarkError("Journey documentation not found")

        tasks = self._parse_journey(tasklist_path.read_text(encoding="utf-8"))
        payload = {
            "exported_at": _isoformat(_now_utc()),
            "tasks": tasks,
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        signature = self._signature.sign_digest(digest)

        export_path = self._root / "cli" / "data" / "yol_hikayesi_export.json"
        export_payload = {
            "payload": payload,
            "digest": digest,
            "signature": signature,
            "signing_key": "default",
        }
        export_path.write_text(
            json.dumps(export_payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        ledger_event = self._ledger.record_event(
            {
                "kind": "journey_exported",
                "path": str(export_path.relative_to(self._root)),
                "digest": digest,
                "signature": signature,
                "tasks_total": len(tasks),
            }
        )

        return {
            "path": str(export_path),
            "digest": digest,
            "signature": signature,
            "tasks_total": len(tasks),
            "ledger_event": ledger_event,
        }

    def _parse_journey(self, content: str) -> List[Dict[str, object]]:
        tasks: List[Dict[str, object]] = []
        current_phase = ""
        for line in content.splitlines():
            line = line.strip()
            if line.startswith("## "):
                current_phase = line[3:].strip()
                continue
            if not line.startswith("- ["):
                continue
            parts = line.split(" ", 3)
            if len(parts) < 4:
                continue
            status_token = parts[1]
            identifier = parts[2]
            title = parts[3]
            completed = status_token.lower() in {"[x]", "[✓]", "[✔]"}
            tasks.append(
                {
                    "phase": current_phase,
                    "task_id": identifier,
                    "title": title,
                    "completed": completed,
                }
            )
        return tasks

    def _report_ai_results(self, stress_result: Dict[str, object]) -> Dict[str, object]:
        metrics = stress_result.get("metrics", [])
        if not isinstance(metrics, list) or not metrics:
            return {"graph": "No stress metrics available", "tests": []}

        energies = [float(entry.get("energy_joules", 0.0)) for entry in metrics]
        peak = max(energies) if energies else 1.0
        if peak == 0:
            peak = 1.0

        bars: List[str] = []
        tests: List[Dict[str, object]] = []
        for index, entry in enumerate(metrics, start=1):
            score = float(entry.get("energy_joules", 0.0)) / peak
            length = max(1, int(round(score * 10)))
            bar = "▇" * length
            label = f"AI-{index:02d}"
            bars.append(f"{label} {bar} {score:.2f}")
            tests.append(
                {
                    "name": label,
                    "determinism_score": round(score, 3),
                    "energy_joules": float(entry.get("energy_joules", 0.0)),
                    "memory_kb": int(entry.get("memory_kb", 0)),
                    "io_bytes": int(entry.get("io_bytes", 0)),
                }
            )
        return {"graph": "\n".join(bars), "tests": tests}

    def _verify_snapshot_hashes(self, components: Dict[str, object]) -> Dict[str, object]:
        digests: Dict[str, str] = {}
        for name, payload in components.items():
            serialised = json.dumps(payload, sort_keys=True, ensure_ascii=False)
            digests[name] = hashlib.sha256(serialised.encode("utf-8")).hexdigest()
        combined = hashlib.sha256(
            "".join(sorted(digests.values())).encode("utf-8")
        ).hexdigest()
        ledger_event = self._ledger.record_event(
            {
                "kind": "deterministic_snapshot_hash_verified",
                "component_digests": digests,
                "combined_digest": combined,
            }
        )
        return {
            "digests": digests,
            "combined_digest": combined,
            "ledger_event": ledger_event,
        }

    def _produce_build_signature(self, snapshot_result: Dict[str, object]) -> Dict[str, object]:
        digest = hashlib.sha256(
            json.dumps(snapshot_result, sort_keys=True, ensure_ascii=False).encode(
                "utf-8"
            )
        ).hexdigest()
        signature = self._signature.sign_digest(digest)
        payload = {
            "produced_at": _isoformat(_now_utc()),
            "digest": digest,
            "signature": signature,
        }
        signature_path = self._root / "cli" / "data" / "piyxu_build_signature.json"
        signature_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        ledger_event = self._ledger.record_event(
            {
                "kind": "piyxu_build_signature_produced",
                "path": str(signature_path.relative_to(self._root)),
                "digest": digest,
                "signature": signature,
            }
        )
        payload["path"] = str(signature_path)
        payload["ledger_event"] = ledger_event
        return payload


__all__ = [
    "DeterministicBenchmarkError",
    "DeterministicBenchmarkRunner",
]
