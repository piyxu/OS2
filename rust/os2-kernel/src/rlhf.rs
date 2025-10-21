use std::collections::{BTreeMap, VecDeque};

use serde::{Deserialize, Serialize};

use crate::metrics::ExecutionMetrics;

pub type ReviewId = u64;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DeterministicMetricSnapshot {
    pub win_rate: f64,
    pub failure_rate: f64,
    pub average_latency: Option<f64>,
}

impl DeterministicMetricSnapshot {
    pub fn from_execution(metrics: &ExecutionMetrics) -> Self {
        let attempts = (metrics.completed_tokens + metrics.failed_tokens) as f64;
        let failure_rate = if attempts == 0.0 {
            0.0
        } else {
            metrics.failed_tokens as f64 / attempts
        };
        Self {
            win_rate: metrics.win_rate,
            failure_rate,
            average_latency: metrics.average_latency,
        }
    }

    pub fn deterministic_score(&self) -> f32 {
        let reliability = self.win_rate - self.failure_rate;
        let latency_penalty = self.average_latency.unwrap_or(0.0) / 1000.0;
        (reliability - latency_penalty).clamp(-1.0, 1.0) as f32
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum PolicyAction {
    Allow,
    RequireHumanReview,
    Block,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PolicyRule {
    pub name: String,
    pub required_tags: Vec<String>,
    pub score_below: Option<f32>,
    pub action: PolicyAction,
}

impl PolicyRule {
    fn matches(&self, input: &InteractionInput) -> bool {
        if !self
            .required_tags
            .iter()
            .all(|tag| input.tags.iter().any(|candidate| candidate == tag))
        {
            return false;
        }
        if let Some(threshold) = self.score_below {
            return input.deterministic_score() <= threshold;
        }
        true
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct InteractionInput {
    pub interaction_id: String,
    pub prompt: String,
    pub response: String,
    pub tags: Vec<String>,
    pub model_score: f32,
    #[serde(default)]
    pub metrics: Option<DeterministicMetricSnapshot>,
}

impl InteractionInput {
    pub fn with_metrics(mut self, metrics: DeterministicMetricSnapshot) -> Self {
        self.metrics = Some(metrics);
        self
    }

    pub fn deterministic_score(&self) -> f32 {
        self.metrics
            .as_ref()
            .map(|snapshot| snapshot.deterministic_score())
            .unwrap_or(self.model_score)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PendingHumanReview {
    pub id: ReviewId,
    pub assigned_reviewer: String,
    pub policy_names: Vec<String>,
    pub input: InteractionInput,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ReviewDecision {
    Approve { notes: Option<String> },
    Reject { reason: String },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum AuditOutcome {
    AutoApproved,
    Blocked {
        policy: String,
    },
    QueuedForReview {
        review_id: ReviewId,
        reviewer: String,
    },
    HumanApproved {
        review_id: ReviewId,
        reviewer: String,
        notes: Option<String>,
    },
    HumanRejected {
        review_id: ReviewId,
        reviewer: String,
        reason: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AuditEntry {
    pub interaction_id: String,
    pub outcome: AuditOutcome,
    pub policies: Vec<String>,
    pub metrics: Option<DeterministicMetricSnapshot>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum PipelineDecision {
    Approved,
    Blocked { policy: String },
    PendingHuman { review_id: ReviewId },
}

pub struct RLHFPipeline {
    policies: Vec<PolicyRule>,
    reviewers: Vec<String>,
    pending_order: VecDeque<ReviewId>,
    pending: BTreeMap<ReviewId, PendingHumanReview>,
    audit_log: Vec<AuditEntry>,
    next_review_id: ReviewId,
    reviewer_cursor: usize,
}

impl RLHFPipeline {
    pub fn new(policies: Vec<PolicyRule>, reviewers: Vec<String>) -> Self {
        Self {
            policies,
            reviewers,
            pending_order: VecDeque::new(),
            pending: BTreeMap::new(),
            audit_log: Vec::new(),
            next_review_id: 1,
            reviewer_cursor: 0,
        }
    }

    fn select_reviewer(&mut self) -> Option<String> {
        if self.reviewers.is_empty() {
            return None;
        }
        let idx = self.reviewer_cursor % self.reviewers.len();
        let reviewer = self.reviewers[idx].clone();
        self.reviewer_cursor = (self.reviewer_cursor + 1) % self.reviewers.len();
        Some(reviewer)
    }

    pub fn submit_interaction(&mut self, input: InteractionInput) -> PipelineDecision {
        let mut matched_policies = Vec::new();
        let mut review_policies = Vec::new();
        let mut block_policy: Option<String> = None;

        for policy in &self.policies {
            if policy.matches(&input) {
                matched_policies.push(policy.name.clone());
                match policy.action {
                    PolicyAction::Allow => {}
                    PolicyAction::RequireHumanReview => review_policies.push(policy.name.clone()),
                    PolicyAction::Block => {
                        block_policy = Some(policy.name.clone());
                        break;
                    }
                }
            }
        }

        if let Some(policy) = block_policy {
            self.audit_log.push(AuditEntry {
                interaction_id: input.interaction_id.clone(),
                outcome: AuditOutcome::Blocked {
                    policy: policy.clone(),
                },
                policies: vec![policy.clone()],
                metrics: input.metrics.clone(),
            });
            return PipelineDecision::Blocked { policy };
        }

        if !review_policies.is_empty() {
            if let Some(reviewer) = self.select_reviewer() {
                let review_id = self.next_review_id;
                self.next_review_id += 1;
                let pending = PendingHumanReview {
                    id: review_id,
                    assigned_reviewer: reviewer.clone(),
                    policy_names: review_policies.clone(),
                    input: input.clone(),
                };
                self.pending.insert(review_id, pending.clone());
                self.pending_order.push_back(review_id);
                self.audit_log.push(AuditEntry {
                    interaction_id: input.interaction_id.clone(),
                    outcome: AuditOutcome::QueuedForReview {
                        review_id,
                        reviewer,
                    },
                    policies: review_policies,
                    metrics: input.metrics.clone(),
                });
                return PipelineDecision::PendingHuman { review_id };
            }

            let reason = format!(
                "requires human review via policies [{}] but no reviewers configured",
                review_policies.join(", ")
            );
            self.audit_log.push(AuditEntry {
                interaction_id: input.interaction_id.clone(),
                outcome: AuditOutcome::Blocked {
                    policy: reason.clone(),
                },
                policies: matched_policies,
                metrics: input.metrics.clone(),
            });
            return PipelineDecision::Blocked { policy: reason };
        }

        self.audit_log.push(AuditEntry {
            interaction_id: input.interaction_id.clone(),
            outcome: AuditOutcome::AutoApproved,
            policies: matched_policies,
            metrics: input.metrics.clone(),
        });
        PipelineDecision::Approved
    }

    pub fn next_review(&self) -> Option<PendingHumanReview> {
        let id = self.pending_order.front()?;
        self.pending.get(id).cloned()
    }

    pub fn pending_reviews(&self) -> Vec<PendingHumanReview> {
        self.pending_order
            .iter()
            .filter_map(|id| self.pending.get(id).cloned())
            .collect()
    }

    pub fn record_feedback(
        &mut self,
        review_id: ReviewId,
        decision: ReviewDecision,
    ) -> Option<AuditOutcome> {
        let pending = self.pending.remove(&review_id)?;
        self.pending_order.retain(|id| id != &review_id);

        let reviewer = pending.assigned_reviewer.clone();
        let policies = pending.policy_names.clone();
        let interaction_id = pending.input.interaction_id.clone();
        let metrics = pending.input.metrics.clone();

        let outcome = match decision {
            ReviewDecision::Approve { notes } => AuditOutcome::HumanApproved {
                review_id,
                reviewer: reviewer.clone(),
                notes,
            },
            ReviewDecision::Reject { reason } => AuditOutcome::HumanRejected {
                review_id,
                reviewer: reviewer.clone(),
                reason,
            },
        };

        self.audit_log.push(AuditEntry {
            interaction_id,
            outcome: outcome.clone(),
            policies,
            metrics,
        });
        Some(outcome)
    }

    pub fn audit_log(&self) -> &[AuditEntry] {
        &self.audit_log
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn require_review_policy() -> PolicyRule {
        PolicyRule {
            name: "pii".into(),
            required_tags: vec!["pii".into()],
            score_below: None,
            action: PolicyAction::RequireHumanReview,
        }
    }

    fn block_policy() -> PolicyRule {
        PolicyRule {
            name: "disallowed".into(),
            required_tags: vec!["banned".into()],
            score_below: None,
            action: PolicyAction::Block,
        }
    }

    fn interaction(id: &str, tags: Vec<&str>, score: f32) -> InteractionInput {
        InteractionInput {
            interaction_id: id.into(),
            prompt: "hello".into(),
            response: "world".into(),
            tags: tags.into_iter().map(|tag| tag.to_string()).collect(),
            model_score: score,
            metrics: None,
        }
    }

    #[test]
    fn auto_approves_when_no_policy_triggers() {
        let mut pipeline = RLHFPipeline::new(vec![], vec!["alice".into()]);
        let decision = pipeline.submit_interaction(interaction("1", vec!["general"], 0.9));
        assert!(matches!(decision, PipelineDecision::Approved));
        assert_eq!(pipeline.audit_log().len(), 1);
        assert!(matches!(
            pipeline.audit_log()[0].outcome,
            AuditOutcome::AutoApproved
        ));
    }

    #[test]
    fn blocks_when_policy_requires_and_no_reviewer() {
        let mut pipeline = RLHFPipeline::new(vec![require_review_policy()], vec![]);
        let decision = pipeline.submit_interaction(interaction("2", vec!["pii"], 0.1));
        match decision {
            PipelineDecision::Blocked { policy } => {
                assert!(policy.contains("requires human review"));
            }
            _ => panic!("expected block"),
        }
    }

    #[test]
    fn queues_and_records_human_feedback() {
        let mut pipeline = RLHFPipeline::new(
            vec![require_review_policy()],
            vec!["alice".into(), "bob".into()],
        );
        let decision = pipeline.submit_interaction(interaction("3", vec!["pii"], 0.2));
        let review_id = match decision {
            PipelineDecision::PendingHuman { review_id } => review_id,
            _ => panic!("expected pending human"),
        };

        let review = pipeline.next_review().expect("pending review");
        assert_eq!(review.assigned_reviewer, "alice");

        let outcome = pipeline
            .record_feedback(review_id, ReviewDecision::Approve { notes: None })
            .expect("feedback recorded");
        match outcome {
            AuditOutcome::HumanApproved { reviewer, .. } => {
                assert_eq!(reviewer, "alice");
            }
            _ => panic!("expected human approved"),
        }

        assert!(pipeline.next_review().is_none());
    }

    #[test]
    fn respects_block_policy() {
        let mut pipeline = RLHFPipeline::new(vec![block_policy()], vec!["alice".into()]);
        let decision = pipeline.submit_interaction(interaction("4", vec!["banned"], 0.0));
        match decision {
            PipelineDecision::Blocked { policy } => {
                assert_eq!(policy, "disallowed");
            }
            _ => panic!("expected blocked decision"),
        }
        assert!(pipeline.pending_reviews().is_empty());
    }

    #[test]
    fn assigns_reviewers_round_robin() {
        let mut pipeline = RLHFPipeline::new(
            vec![require_review_policy()],
            vec!["alice".into(), "bob".into()],
        );
        let first = pipeline.submit_interaction(interaction("5", vec!["pii"], 0.1));
        let second = pipeline.submit_interaction(interaction("6", vec!["pii"], 0.1));

        let first_id = match first {
            PipelineDecision::PendingHuman { review_id } => review_id,
            _ => panic!("expected pending human"),
        };
        let second_id = match second {
            PipelineDecision::PendingHuman { review_id } => review_id,
            _ => panic!("expected pending human"),
        };

        let pending = pipeline.pending_reviews();
        assert_eq!(pending.len(), 2);
        assert_eq!(pending[0].assigned_reviewer, "alice");
        assert_eq!(pending[1].assigned_reviewer, "bob");

        pipeline
            .record_feedback(first_id, ReviewDecision::Approve { notes: None })
            .unwrap();
        pipeline
            .record_feedback(
                second_id,
                ReviewDecision::Reject {
                    reason: "policy".into(),
                },
            )
            .unwrap();

        assert!(pipeline.pending_reviews().is_empty());
        assert_eq!(pipeline.audit_log().len(), 4);
    }

    #[test]
    fn deterministic_metrics_drive_policy_decisions() {
        let policy = PolicyRule {
            name: "quality".into(),
            required_tags: vec!["prod".into()],
            score_below: Some(0.4),
            action: PolicyAction::Block,
        };
        let mut pipeline = RLHFPipeline::new(vec![policy], vec![]);

        let metrics = DeterministicMetricSnapshot {
            win_rate: 0.3,
            failure_rate: 0.4,
            average_latency: Some(50.0),
        };

        let input = interaction("7", vec!["prod"], 0.9).with_metrics(metrics);
        let decision = pipeline.submit_interaction(input);
        match decision {
            PipelineDecision::Blocked { policy } => assert_eq!(policy, "quality"),
            other => panic!("expected deterministic block, got {:?}", other),
        }
    }
}
