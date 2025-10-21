"""Policy-driven RLHF pipeline utilities."""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Deque, Dict, List, Optional, Sequence


class PolicyAction(str, Enum):
    ALLOW = "allow"
    REQUIRE_REVIEW = "require_review"
    BLOCK = "block"


@dataclass
class PolicyRule:
    name: str
    required_tags: Sequence[str] = field(default_factory=list)
    score_below: Optional[float] = None
    action: PolicyAction = PolicyAction.ALLOW

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "PolicyRule":
        return cls(
            name=str(payload.get("name", "policy")),
            required_tags=list(payload.get("required_tags", []) or []),
            score_below=float(payload["score_below"]) if payload.get("score_below") is not None else None,
            action=PolicyAction(payload.get("action", "allow")),
        )

    def matches(self, interaction: "InteractionInput") -> bool:
        for tag in self.required_tags:
            if tag not in interaction.tags:
                return False
        if self.score_below is not None:
            if interaction.deterministic_score() > self.score_below:
                return False
        return True


@dataclass
class DeterministicMetricSnapshot:
    win_rate: float
    failure_rate: float
    average_latency: Optional[float]

    def deterministic_score(self) -> float:
        reliability = self.win_rate - self.failure_rate
        latency_penalty = (self.average_latency or 0.0) / 1000.0
        return max(min(reliability - latency_penalty, 1.0), -1.0)


@dataclass
class InteractionInput:
    interaction_id: str
    prompt: str
    response: str
    tags: Sequence[str]
    model_score: float
    metrics: Optional[DeterministicMetricSnapshot] = None

    def deterministic_score(self) -> float:
        if self.metrics:
            return self.metrics.deterministic_score()
        return self.model_score


@dataclass
class PipelineDecision:
    status: str
    reason: Optional[str] = None
    review_id: Optional[int] = None
    reviewer: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        return {
            "status": self.status,
            "reason": self.reason,
            "review_id": self.review_id,
            "reviewer": self.reviewer,
        }


@dataclass
class AuditEntry:
    interaction_id: str
    outcome: PipelineDecision
    triggered_policies: List[str]

    def to_dict(self) -> Dict[str, object]:
        return {
            "interaction_id": self.interaction_id,
            "outcome": self.outcome.to_dict(),
            "triggered_policies": list(self.triggered_policies),
        }


class RLHFPipeline:
    def __init__(self, policies: Sequence[PolicyRule], reviewers: Sequence[str]) -> None:
        self._policies = list(policies)
        self._reviewers = list(reviewers) or ["reviewer"]
        self._reviewer_queue: Deque[str] = deque(self._reviewers)
        self._next_review_id = 1
        self._audit_log: List[AuditEntry] = []

    @classmethod
    def from_policy_file(cls, path: Path) -> "RLHFPipeline":
        if not path.exists():
            return cls(policies=[], reviewers=["reviewer"])
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return cls(policies=[], reviewers=["reviewer"])
        policies = [PolicyRule.from_dict(entry) for entry in payload.get("policies", [])]
        reviewers = payload.get("reviewers", []) or ["reviewer"]
        return cls(policies=policies, reviewers=reviewers)

    def submit_interaction(self, interaction: InteractionInput) -> PipelineDecision:
        triggered: List[str] = []
        pending_review = False
        block_policy: Optional[str] = None

        for policy in self._policies:
            if not policy.matches(interaction):
                continue
            triggered.append(policy.name)
            if policy.action == PolicyAction.BLOCK:
                block_policy = policy.name
                break
            if policy.action == PolicyAction.REQUIRE_REVIEW:
                pending_review = True

        if block_policy:
            decision = PipelineDecision(status="blocked", reason=block_policy)
        elif pending_review:
            reviewer = self._next_reviewer()
            review_id = self._next_review_id
            self._next_review_id += 1
            decision = PipelineDecision(status="pending_review", review_id=review_id, reviewer=reviewer, reason="policy_review")
        else:
            decision = PipelineDecision(status="approved")

        self._audit_log.append(AuditEntry(interaction.interaction_id, decision, triggered))
        return decision

    def _next_reviewer(self) -> str:
        chosen = self._reviewer_queue[0]
        self._reviewer_queue.rotate(-1)
        return chosen

    def audit_log(self) -> List[Dict[str, object]]:
        return [entry.to_dict() for entry in self._audit_log]


__all__ = [
    "PolicyAction",
    "PolicyRule",
    "DeterministicMetricSnapshot",
    "InteractionInput",
    "PipelineDecision",
    "RLHFPipeline",
]
