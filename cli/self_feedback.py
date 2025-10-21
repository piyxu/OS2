"""Self-feedback analyzer that processes user interaction transcripts."""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .snapshot_ledger import SnapshotLedger, SnapshotLedgerError


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _isoformat_utc(dt: datetime) -> str:
    return dt.isoformat().replace("+00:00", "Z")


@dataclass
class Interaction:
    """Normalized representation of a single transcript interaction."""

    command: str
    status: int
    timestamp: str
    sentiment: str
    stdout_length: int
    stderr_length: int
    friction_score: int
    args: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "command": self.command,
            "status": self.status,
            "timestamp": self.timestamp,
            "sentiment": self.sentiment,
            "stdout_length": self.stdout_length,
            "stderr_length": self.stderr_length,
            "friction_score": self.friction_score,
            "args": list(self.args),
        }


class SelfFeedbackError(RuntimeError):
    """Raised when the self-feedback analyzer cannot persist results."""


class SelfFeedbackAnalyzer:
    """Processes transcript entries and summarizes interaction quality."""

    def __init__(
        self,
        root: Path,
        ledger: SnapshotLedger,
        *,
        max_history: int = 50,
    ) -> None:
        self._ledger = ledger
        self._path = root / "cli" / "data" / "self_feedback.json"
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._max_history = max(1, max_history)
        self._lock = threading.RLock()
        self._state = self._load_state()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def ingest(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        """Record a transcript *entry* and update aggregated feedback metrics."""

        with self._lock:
            interaction = self._normalize_entry(entry)
            self._apply_interaction(interaction)
            summary = self._build_summary_locked()
            event_payload = {
                "kind": "self_feedback_interaction_recorded",
                "command": interaction.command,
                "status": interaction.status,
                "sentiment": interaction.sentiment,
                "timestamp": interaction.timestamp,
                "friction_score": interaction.friction_score,
                "engagement_score": summary["engagement_score"],
                "totals": {
                    "total": summary["total_interactions"],
                    "positive": summary["positive_count"],
                    "negative": summary["negative_count"],
                    "neutral": summary["neutral_count"],
                },
                "command_stats": summary["commands"].get(interaction.command, {}),
            }
            try:
                ledger_event = self._ledger.record_event(event_payload)
            except SnapshotLedgerError as exc:  # pragma: no cover - defensive
                raise SelfFeedbackError(str(exc)) from exc
            self._save_locked()
            return {
                "interaction": interaction.to_dict(),
                "summary": summary,
                "ledger_event": ledger_event,
            }

    def summary(self) -> Dict[str, Any]:
        with self._lock:
            return self._build_summary_locked()

    def recent_interactions(self, limit: int = 10) -> List[Dict[str, Any]]:
        if limit <= 0:
            return []
        with self._lock:
            history = self._state["recent"]
            return [dict(entry) for entry in history[:limit]]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _load_state(self) -> Dict[str, Any]:
        state = {
            "total_interactions": 0,
            "success_count": 0,
            "failure_count": 0,
            "positive_count": 0,
            "negative_count": 0,
            "neutral_count": 0,
            "engagement_score": 0,
            "commands": {},
            "recent": [],
            "last_interaction_at": None,
        }
        if not self._path.exists():
            return state
        try:
            payload = json.loads(self._path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return state
        if not isinstance(payload, dict):
            return state
        for key in (
            "total_interactions",
            "success_count",
            "failure_count",
            "positive_count",
            "negative_count",
            "neutral_count",
            "engagement_score",
            "last_interaction_at",
        ):
            if key in payload:
                state[key] = payload[key]
        commands = payload.get("commands", {})
        if isinstance(commands, dict):
            normalized_commands: Dict[str, Dict[str, Any]] = {}
            for name, stats in commands.items():
                if isinstance(stats, dict):
                    normalized_commands[str(name)] = {
                        "count": int(stats.get("count", 0) or 0),
                        "success": int(stats.get("success", 0) or 0),
                        "failure": int(stats.get("failure", 0) or 0),
                        "positive": int(stats.get("positive", 0) or 0),
                        "negative": int(stats.get("negative", 0) or 0),
                        "neutral": int(stats.get("neutral", 0) or 0),
                        "friction_events": int(stats.get("friction_events", 0) or 0),
                    }
            state["commands"] = normalized_commands
        history = payload.get("recent", [])
        if isinstance(history, list):
            normalized_history = []
            for item in history:
                if isinstance(item, dict):
                    normalized_history.append(dict(item))
            state["recent"] = normalized_history[: self._max_history]
        return state

    def _normalize_entry(self, entry: Dict[str, Any]) -> Interaction:
        command = str(entry.get("command", "unknown"))
        status = int(entry.get("status", 0) or 0)
        timestamp = entry.get("timestamp") or _isoformat_utc(_now_utc())
        stdout = str(entry.get("stdout", "") or "")
        stderr = str(entry.get("stderr", "") or "")
        args_value = entry.get("args")
        args: List[str]
        if isinstance(args_value, list):
            args = [str(arg) for arg in args_value]
        else:
            args = []
        sentiment = self._detect_sentiment(status, stdout, stderr)
        friction = self._estimate_friction(status, stderr)
        return Interaction(
            command=command,
            status=status,
            timestamp=str(timestamp),
            sentiment=sentiment,
            stdout_length=len(stdout.strip()),
            stderr_length=len(stderr.strip()),
            friction_score=friction,
            args=args,
        )

    def _detect_sentiment(self, status: int, stdout: str, stderr: str) -> str:
        if status == 0 and not stderr.strip():
            return "positive"
        if status == 0:
            return "neutral"
        if stderr.strip():
            return "negative"
        return "neutral"

    def _estimate_friction(self, status: int, stderr: str) -> int:
        if status == 0 and not stderr.strip():
            return 0
        lines = [line for line in stderr.splitlines() if line.strip()]
        if not lines:
            return 1 if status != 0 else 0
        return min(10, len(lines) + (2 if status != 0 else 0))

    def _apply_interaction(self, interaction: Interaction) -> None:
        state = self._state
        state["total_interactions"] += 1
        if interaction.status == 0:
            state["success_count"] += 1
        else:
            state["failure_count"] += 1
        sentiment = interaction.sentiment
        state[f"{sentiment}_count"] += 1
        if sentiment == "positive":
            state["engagement_score"] += 1
        elif sentiment == "negative":
            state["engagement_score"] -= 1
        commands = state.setdefault("commands", {})
        stats = commands.setdefault(
            interaction.command,
            {
                "count": 0,
                "success": 0,
                "failure": 0,
                "positive": 0,
                "negative": 0,
                "neutral": 0,
                "friction_events": 0,
            },
        )
        stats["count"] += 1
        if interaction.status == 0:
            stats["success"] += 1
        else:
            stats["failure"] += 1
        stats[sentiment] += 1
        if interaction.friction_score:
            stats["friction_events"] += 1
        history: List[Dict[str, Any]] = state.setdefault("recent", [])
        history.insert(0, interaction.to_dict())
        del history[self._max_history :]
        state["last_interaction_at"] = interaction.timestamp

    def _build_summary_locked(self) -> Dict[str, Any]:
        state = self._state
        total = state["total_interactions"]
        success = state["success_count"]
        failure = state["failure_count"]
        positive = state["positive_count"]
        negative = state["negative_count"]
        neutral = state["neutral_count"]
        engagement = state["engagement_score"]
        success_rate = success / total if total else 0.0
        error_rate = failure / total if total else 0.0
        sentiment_score = (positive - negative) / total if total else 0.0
        positive_rate = positive / total if total else 0.0
        commands = {
            name: dict(stats) for name, stats in state.get("commands", {}).items()
        }
        top_commands = self._top_commands(commands)
        high_friction = [
            {
                "command": name,
                "friction_events": stats.get("friction_events", 0),
                "count": stats.get("count", 0),
            }
            for name, stats in commands.items()
            if stats.get("friction_events", 0) > 0
        ]
        high_friction.sort(
            key=lambda item: (item["friction_events"], item["count"], item["command"]),
            reverse=True,
        )
        return {
            "total_interactions": total,
            "success_count": success,
            "failure_count": failure,
            "positive_count": positive,
            "negative_count": negative,
            "neutral_count": neutral,
            "success_rate": success_rate,
            "error_rate": error_rate,
            "positive_rate": positive_rate,
            "sentiment_score": sentiment_score,
            "engagement_score": engagement,
            "last_interaction_at": state.get("last_interaction_at"),
            "commands": commands,
            "top_commands": top_commands,
            "high_friction_commands": high_friction[:5],
        }

    def _top_commands(self, commands: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        items: List[Tuple[str, Dict[str, Any]]] = list(commands.items())
        items.sort(key=lambda item: (item[1].get("count", 0), item[0]), reverse=True)
        top: List[Dict[str, Any]] = []
        for name, stats in items[:5]:
            count = stats.get("count", 0)
            success = stats.get("success", 0)
            success_rate = success / count if count else 0.0
            top.append(
                {
                    "command": name,
                    "count": count,
                    "success_rate": success_rate,
                    "positive": stats.get("positive", 0),
                    "negative": stats.get("negative", 0),
                }
            )
        return top

    def _save_locked(self) -> None:
        try:
            serialized = json.dumps(self._state, indent=2, ensure_ascii=False)
            self._path.write_text(serialized + "\n", encoding="utf-8")
        except OSError as exc:  # pragma: no cover - defensive
            raise SelfFeedbackError(str(exc)) from exc


__all__ = ["SelfFeedbackAnalyzer", "SelfFeedbackError"]
