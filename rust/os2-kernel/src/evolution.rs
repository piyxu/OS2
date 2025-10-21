use std::collections::VecDeque;

use serde::{Deserialize, Serialize};

use crate::metrics::ExecutionMetrics;
use crate::snapshot::{SnapshotEngine, SnapshotId};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EvolverProposal {
    pub id: u64,
    pub summary: String,
    pub prompt_delta: Option<String>,
    pub policy_delta: Option<serde_json::Value>,
    pub rationale: String,
}

impl EvolverProposal {
    pub fn new(id: u64, summary: impl Into<String>) -> Self {
        Self {
            id,
            summary: summary.into(),
            prompt_delta: None,
            policy_delta: None,
            rationale: String::new(),
        }
    }

    pub fn with_prompt_delta(mut self, delta: impl Into<String>) -> Self {
        self.prompt_delta = Some(delta.into());
        self
    }

    pub fn with_policy_delta(mut self, delta: serde_json::Value) -> Self {
        self.policy_delta = Some(delta);
        self
    }

    pub fn with_rationale(mut self, rationale: impl Into<String>) -> Self {
        self.rationale = rationale.into();
        self
    }
}

#[derive(Debug, Clone)]
pub struct EvolverAgent {
    next_id: u64,
    win_rate_target: f64,
    latency_target: f64,
    failure_rate_limit: f64,
    history: VecDeque<LearningSample>,
    history_limit: usize,
    learning_rate: f64,
}

impl Default for EvolverAgent {
    fn default() -> Self {
        Self {
            next_id: 1,
            win_rate_target: 0.9,
            latency_target: 25.0,
            failure_rate_limit: 0.2,
            history: VecDeque::with_capacity(16),
            history_limit: 64,
            learning_rate: 0.15,
        }
    }
}

impl EvolverAgent {
    pub fn new(win_rate_target: f64, latency_target: f64, failure_rate_limit: f64) -> Self {
        Self {
            next_id: 1,
            win_rate_target,
            latency_target,
            failure_rate_limit,
            history: VecDeque::with_capacity(16),
            history_limit: 64,
            learning_rate: 0.15,
        }
    }

    pub fn with_history_limit(mut self, limit: usize) -> Self {
        self.history_limit = limit.max(4);
        self
    }

    pub fn with_learning_rate(mut self, rate: f64) -> Self {
        self.learning_rate = rate.clamp(0.01, 1.0);
        self
    }

    pub fn analyze(&mut self, metrics: &ExecutionMetrics) -> Vec<EvolverProposal> {
        self.observe(metrics);
        self.build_proposals(metrics)
    }

    pub fn analyze_from_observation(&mut self, metrics: &ExecutionMetrics) -> Vec<EvolverProposal> {
        self.build_proposals(metrics)
    }

    fn build_proposals(&mut self, metrics: &ExecutionMetrics) -> Vec<EvolverProposal> {
        let mut proposals = Vec::new();
        let failure_rate = failure_rate(metrics);

        if metrics.win_rate < self.win_rate_target {
            let proposal =
                EvolverProposal::new(self.next_id, "Tighten reasoning prompt for accuracy")
                    .with_prompt_delta("Add explicit verification checklist before final answers.")
                    .with_rationale(format!(
                        "Observed win rate {:.1}% is below target {:.1}%.",
                        metrics.win_rate * 100.0,
                        self.win_rate_target * 100.0
                    ));
            self.next_id += 1;
            proposals.push(proposal);
        }

        if metrics
            .average_latency
            .map(|latency| latency > self.latency_target)
            .unwrap_or(false)
        {
            let proposal = EvolverProposal::new(self.next_id, "Constrain planning stage latency")
                .with_policy_delta(serde_json::json!({
                    "plan": {"max_latency": self.latency_target}
                }))
                .with_rationale(format!(
                    "Average latency {:.2} exceeds {:.2} threshold.",
                    metrics.average_latency.unwrap(),
                    self.latency_target
                ));
            self.next_id += 1;
            proposals.push(proposal);
        }

        if failure_rate > self.failure_rate_limit {
            let proposal =
                EvolverProposal::new(self.next_id, "Increase guardrails for failing actions")
                    .with_policy_delta(serde_json::json!({
                        "safety": {"max_failure_rate": self.failure_rate_limit}
                    }))
                    .with_rationale(format!(
                        "Failure rate {:.1}% exceeds cap {:.1}%.",
                        failure_rate * 100.0,
                        self.failure_rate_limit * 100.0
                    ));
            self.next_id += 1;
            proposals.push(proposal);
        }

        proposals
    }

    pub fn observe(&mut self, metrics: &ExecutionMetrics) {
        let sample = LearningSample::from_metrics(metrics);
        if self.history.len() == self.history_limit {
            self.history.pop_front();
        }

        if let Some(baseline) = self.history.back() {
            let learning_rate = self.learning_rate;
            let delta_win = sample.win_rate - baseline.win_rate;
            if delta_win.abs() > 0.01 {
                let adjustment = delta_win * learning_rate;
                self.win_rate_target = (self.win_rate_target + adjustment).clamp(0.5, 0.99);
            }

            if let (Some(latency), Some(prev_latency)) =
                (sample.average_latency, baseline.average_latency)
            {
                let delta_latency = latency - prev_latency;
                if delta_latency.abs() > 0.01 {
                    let adjustment = delta_latency * learning_rate;
                    self.latency_target = (self.latency_target + adjustment).clamp(1.0, 10_000.0);
                }
            }

            let delta_failure = sample.failure_rate - baseline.failure_rate;
            if delta_failure.abs() > 0.001 {
                let adjustment = delta_failure * learning_rate;
                self.failure_rate_limit = (self.failure_rate_limit + adjustment).clamp(0.01, 0.9);
            }
        }

        self.history.push_back(sample);
    }

    pub fn learning_summary(&self) -> LearningSummary {
        let window: Vec<_> = self.history.iter().rev().take(8).cloned().collect();
        LearningSummary {
            win_rate_target: self.win_rate_target,
            latency_target: self.latency_target,
            failure_rate_limit: self.failure_rate_limit,
            recent: window,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LearningSample {
    pub win_rate: f64,
    pub average_latency: Option<f64>,
    pub failure_rate: f64,
}

impl LearningSample {
    pub fn from_metrics(metrics: &ExecutionMetrics) -> Self {
        Self {
            win_rate: metrics.win_rate,
            average_latency: metrics.average_latency,
            failure_rate: failure_rate(metrics),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LearningSummary {
    pub win_rate_target: f64,
    pub latency_target: f64,
    pub failure_rate_limit: f64,
    pub recent: Vec<LearningSample>,
}

#[derive(Debug, Clone)]
pub enum SafetyRule {
    MinWinRate { absolute: f64 },
    MaxFailureRate { ratio: f64 },
    MaxLatencyIncrease { max_increase: f64 },
}

impl SafetyRule {
    fn name(&self) -> &'static str {
        match self {
            SafetyRule::MinWinRate { .. } => "min_win_rate",
            SafetyRule::MaxFailureRate { .. } => "max_failure_rate",
            SafetyRule::MaxLatencyIncrease { .. } => "max_latency_increase",
        }
    }

    pub fn evaluate(
        &self,
        baseline: &ExecutionMetrics,
        candidate: &ExecutionMetrics,
    ) -> SafetyOutcome {
        match self {
            SafetyRule::MinWinRate { absolute } => {
                let passed = candidate.win_rate >= *absolute;
                let detail = format!(
                    "candidate_win_rate={:.3} baseline_win_rate={:.3} threshold={:.3}",
                    candidate.win_rate, baseline.win_rate, absolute
                );
                SafetyOutcome::new(self.name(), passed, detail)
            }
            SafetyRule::MaxFailureRate { ratio } => {
                let candidate_fail = failure_rate(candidate);
                let passed = candidate_fail <= *ratio;
                let detail = format!(
                    "candidate_failure_rate={:.3} baseline_failure_rate={:.3} threshold={:.3}",
                    candidate_fail,
                    failure_rate(baseline),
                    ratio
                );
                SafetyOutcome::new(self.name(), passed, detail)
            }
            SafetyRule::MaxLatencyIncrease { max_increase } => {
                let baseline_latency = baseline.average_latency.unwrap_or(0.0);
                let candidate_latency = candidate.average_latency.unwrap_or(0.0);
                let passed = candidate_latency <= baseline_latency + max_increase;
                let detail = format!(
                    "candidate_latency={:.3} baseline_latency={:.3} max_increase={:.3}",
                    candidate_latency, baseline_latency, max_increase
                );
                SafetyOutcome::new(self.name(), passed, detail)
            }
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct SafetyOutcome {
    pub rule: String,
    pub passed: bool,
    pub detail: String,
}

impl SafetyOutcome {
    fn new(rule: &str, passed: bool, detail: String) -> Self {
        Self {
            rule: rule.to_string(),
            passed,
            detail,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct EvaluationReport {
    pub label: String,
    pub proposal: EvolverProposal,
    pub baseline_snapshot: SnapshotId,
    pub candidate_snapshot: SnapshotId,
    pub baseline_metrics: ExecutionMetrics,
    pub candidate_metrics: ExecutionMetrics,
    pub win_rate_delta: f64,
    pub safety_outcomes: Vec<SafetyOutcome>,
    pub promoted: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationRunResult {
    pub metrics: ExecutionMetrics,
    pub state: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelfEvolutionReport {
    pub metrics: ExecutionMetrics,
    pub proposals: Vec<EvolverProposal>,
    pub triggered: bool,
}

#[derive(Debug)]
pub struct SelfEvolutionTrigger {
    agent: EvolverAgent,
    window: VecDeque<ExecutionMetrics>,
    window_limit: usize,
    degrade_threshold: f64,
    pending: Vec<SelfEvolutionReport>,
}

impl SelfEvolutionTrigger {
    pub fn new(agent: EvolverAgent) -> Self {
        Self {
            agent,
            window: VecDeque::with_capacity(8),
            window_limit: 32,
            degrade_threshold: 0.05,
            pending: Vec::new(),
        }
    }

    pub fn with_degrade_threshold(mut self, threshold: f64) -> Self {
        self.degrade_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    pub fn observe(
        &mut self,
        events: &[crate::event_bus::KernelEvent],
    ) -> Option<SelfEvolutionReport> {
        if events.is_empty() {
            return None;
        }

        let metrics = ExecutionMetrics::from_events(events);
        let historical_avg =
            self.window.iter().map(|m| m.win_rate).sum::<f64>() / self.window.len().max(1) as f64;
        let failure_avg = self.window.iter().map(|m| failure_rate(m)).sum::<f64>()
            / self.window.len().max(1) as f64;

        if self.window.len() == self.window_limit {
            self.window.pop_front();
        }
        self.window.push_back(metrics.clone());

        self.agent.observe(&metrics);

        let win_rate_drop = if self.window.len() > 1 {
            (historical_avg - metrics.win_rate).max(0.0)
        } else {
            0.0
        };
        let failure_spike = if self.window.len() > 1 {
            (failure_rate(&metrics) - failure_avg).max(0.0)
        } else {
            0.0
        };

        let mut triggered = win_rate_drop >= self.degrade_threshold || failure_spike >= 0.02;
        let proposals = if triggered {
            self.agent.analyze_from_observation(&metrics)
        } else {
            Vec::new()
        };

        if proposals.is_empty() {
            triggered = false;
        }

        let report = SelfEvolutionReport {
            metrics: metrics.clone(),
            proposals,
            triggered,
        };

        if triggered {
            self.pending.push(report.clone());
            Some(report)
        } else {
            None
        }
    }

    pub fn take_pending(&mut self) -> Vec<SelfEvolutionReport> {
        std::mem::take(&mut self.pending)
    }

    pub fn agent(&self) -> &EvolverAgent {
        &self.agent
    }

    pub fn agent_mut(&mut self) -> &mut EvolverAgent {
        &mut self.agent
    }
}

#[derive(Debug)]
pub struct SnapshotEvaluator {
    engine: SnapshotEngine,
    minimum_win_rate_delta: f64,
    safety_rules: Vec<SafetyRule>,
}

impl SnapshotEvaluator {
    pub fn new(minimum_win_rate_delta: f64, safety_rules: Vec<SafetyRule>) -> Self {
        Self {
            engine: SnapshotEngine::new(),
            minimum_win_rate_delta,
            safety_rules,
        }
    }

    pub fn evaluate_proposal<FB, FC>(
        &mut self,
        label: impl Into<String>,
        proposal: &EvolverProposal,
        baseline: FB,
        candidate: FC,
    ) -> EvaluationReport
    where
        FB: FnOnce() -> EvaluationRunResult,
        FC: FnOnce(serde_json::Value) -> EvaluationRunResult,
    {
        let label = label.into();
        let baseline_result = baseline();
        let baseline_snapshot = self.engine.checkpoint(
            format!("{label}-baseline"),
            serde_json::json!({
                "proposal_id": proposal.id,
                "label": label,
                "stage": "baseline",
                "state": baseline_result.state,
            }),
        );
        let baseline_state = self
            .engine
            .restore(baseline_snapshot)
            .and_then(|snapshot| snapshot.state.get("state").cloned())
            .unwrap_or(serde_json::Value::Null);

        let candidate_result = candidate(baseline_state);
        let candidate_snapshot = self.engine.checkpoint(
            format!("{label}-candidate"),
            serde_json::json!({
                "proposal_id": proposal.id,
                "label": label,
                "stage": "candidate",
                "state": candidate_result.state,
            }),
        );

        let win_rate_delta = candidate_result.metrics.win_rate - baseline_result.metrics.win_rate;
        let mut promoted = win_rate_delta >= self.minimum_win_rate_delta
            && candidate_result.metrics.win_rate >= baseline_result.metrics.win_rate;

        let mut safety_outcomes = Vec::with_capacity(self.safety_rules.len());
        for rule in &self.safety_rules {
            let outcome = rule.evaluate(&baseline_result.metrics, &candidate_result.metrics);
            if !outcome.passed {
                promoted = false;
            }
            safety_outcomes.push(outcome);
        }

        EvaluationReport {
            label,
            proposal: proposal.clone(),
            baseline_snapshot,
            candidate_snapshot,
            baseline_metrics: baseline_result.metrics,
            candidate_metrics: candidate_result.metrics,
            win_rate_delta,
            safety_outcomes,
            promoted,
        }
    }
}

fn failure_rate(metrics: &ExecutionMetrics) -> f64 {
    let attempts = (metrics.completed_tokens + metrics.failed_tokens) as f64;
    if attempts == 0.0 {
        0.0
    } else {
        metrics.failed_tokens as f64 / attempts
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::event_bus::{EventKind, KernelEvent};
    use crate::scheduler::{TokenId, TokenKind};

    fn baseline_metrics() -> ExecutionMetrics {
        ExecutionMetrics {
            total_tokens: 10,
            completed_tokens: 7,
            failed_tokens: 3,
            win_rate: 0.7,
            average_latency: Some(24.0),
            ..ExecutionMetrics::default()
        }
    }

    #[test]
    fn evolver_agent_generates_proposals_based_on_metrics() {
        let mut agent = EvolverAgent::new(0.9, 20.0, 0.2);
        let metrics = baseline_metrics();
        let proposals = agent.analyze(&metrics);

        assert!(
            proposals
                .iter()
                .any(|proposal| proposal.prompt_delta.is_some())
        );
        assert!(
            proposals
                .iter()
                .any(|proposal| proposal.policy_delta.is_some())
        );
        assert!(
            proposals
                .iter()
                .any(|proposal| !proposal.rationale.is_empty())
        );
    }

    #[test]
    fn snapshot_evaluator_promotes_candidate_when_rules_pass() {
        let mut evaluator = SnapshotEvaluator::new(
            0.05,
            vec![
                SafetyRule::MinWinRate { absolute: 0.75 },
                SafetyRule::MaxFailureRate { ratio: 0.2 },
                SafetyRule::MaxLatencyIncrease { max_increase: 5.0 },
            ],
        );
        let mut agent = EvolverAgent::default();
        let proposal = agent.analyze(&baseline_metrics())[0].clone();

        let baseline_run = || EvaluationRunResult {
            metrics: baseline_metrics(),
            state: serde_json::json!({"prompt": "baseline"}),
        };

        let candidate_metrics = ExecutionMetrics {
            total_tokens: 10,
            completed_tokens: 9,
            failed_tokens: 1,
            win_rate: 0.9,
            average_latency: Some(22.0),
            ..ExecutionMetrics::default()
        };

        let report =
            evaluator.evaluate_proposal("policy_tuning", &proposal, baseline_run, |state| {
                assert_eq!(state["prompt"], "baseline");
                EvaluationRunResult {
                    metrics: candidate_metrics,
                    state: serde_json::json!({"prompt": "candidate"}),
                }
            });

        assert!(report.promoted);
        assert!(report.win_rate_delta > 0.15);
        assert!(report.safety_outcomes.iter().all(|outcome| outcome.passed));
        assert_ne!(report.baseline_snapshot, report.candidate_snapshot);
    }

    #[test]
    fn snapshot_evaluator_rejects_candidate_on_safety_violation() {
        let mut evaluator = SnapshotEvaluator::new(
            0.01,
            vec![
                SafetyRule::MaxLatencyIncrease { max_increase: 1.0 },
                SafetyRule::MaxFailureRate { ratio: 0.2 },
            ],
        );

        let proposal = EvolverProposal::new(1, "Reduce cost")
            .with_policy_delta(serde_json::json!({"plan": {"max_latency": 10.0}}));

        let baseline_run = || EvaluationRunResult {
            metrics: baseline_metrics(),
            state: serde_json::json!({"prompt": "baseline"}),
        };

        let candidate_metrics = ExecutionMetrics {
            total_tokens: 10,
            completed_tokens: 8,
            failed_tokens: 2,
            win_rate: 0.8,
            average_latency: Some(30.0),
            ..ExecutionMetrics::default()
        };

        let report =
            evaluator.evaluate_proposal("latency_regression", &proposal, baseline_run, |_| {
                EvaluationRunResult {
                    metrics: candidate_metrics,
                    state: serde_json::json!({"prompt": "candidate"}),
                }
            });

        assert!(!report.promoted);
        assert!(report.safety_outcomes.iter().any(|outcome| !outcome.passed));
    }

    #[test]
    fn agent_learns_from_metrics_trend() {
        let mut agent = EvolverAgent::default().with_learning_rate(0.5);
        let mut metrics = baseline_metrics();
        metrics.win_rate = 0.75;
        agent.observe(&metrics);
        metrics.win_rate = 0.85;
        agent.observe(&metrics);

        let summary = agent.learning_summary();
        assert!(summary.win_rate_target > 0.9 - 1e-6);
    }

    #[test]
    fn self_evolution_trigger_detects_drop() {
        fn success_events(token_id: u64) -> Vec<KernelEvent> {
            vec![
                KernelEvent {
                    token_id: TokenId::new(token_id),
                    kind: EventKind::Scheduled,
                    detail: serde_json::json!({"kind": TokenKind::Plan}),
                    timestamp: token_id * 3,
                },
                KernelEvent {
                    token_id: TokenId::new(token_id),
                    kind: EventKind::Started,
                    detail: serde_json::json!({}),
                    timestamp: token_id * 3 + 1,
                },
                KernelEvent {
                    token_id: TokenId::new(token_id),
                    kind: EventKind::Completed,
                    detail: serde_json::json!({}),
                    timestamp: token_id * 3 + 2,
                },
            ]
        }

        fn failure_events(token_id: u64) -> Vec<KernelEvent> {
            vec![
                KernelEvent {
                    token_id: TokenId::new(token_id),
                    kind: EventKind::Scheduled,
                    detail: serde_json::json!({"kind": TokenKind::Plan}),
                    timestamp: token_id * 4,
                },
                KernelEvent {
                    token_id: TokenId::new(token_id),
                    kind: EventKind::Started,
                    detail: serde_json::json!({}),
                    timestamp: token_id * 4 + 1,
                },
                KernelEvent {
                    token_id: TokenId::new(token_id),
                    kind: EventKind::CapabilityViolation,
                    detail: serde_json::json!({"error": "budget"}),
                    timestamp: token_id * 4 + 2,
                },
            ]
        }

        let mut trigger =
            SelfEvolutionTrigger::new(EvolverAgent::default()).with_degrade_threshold(0.04);

        let baseline = success_events(1);
        trigger.observe(&baseline);

        let degraded = failure_events(2);
        let report = trigger.observe(&degraded);
        assert!(report.map(|r| r.triggered).unwrap_or(false));
    }
}
