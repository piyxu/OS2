use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::event_bus::{EventKind, KernelEvent};
use crate::resource::ResourceUsageSnapshot;
use crate::scheduler::{TokenId, TokenKind};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ExecutionMetrics {
    pub total_tokens: usize,
    pub completed_tokens: usize,
    pub failed_tokens: usize,
    pub win_rate: f64,
    pub average_latency: Option<f64>,
    pub token_runs: BTreeMap<TokenId, TokenRunMetrics>,
    pub per_kind: BTreeMap<TokenKind, TokenKindMetrics>,
    pub capability_usage: BTreeMap<String, CapabilityUsageMetrics>,
    pub resource_usage: Option<ResourceUsageSnapshot>,
}

impl ExecutionMetrics {
    pub fn from_events(events: &[KernelEvent]) -> Self {
        MetricsBuilder::default().collect(events)
    }

    pub fn render_report(&self) -> String {
        let mut lines = Vec::new();
        lines.push(format!(
            "Metrics: total_tokens={} completed={} failed={} win_rate={:.1}%",
            self.total_tokens,
            self.completed_tokens,
            self.failed_tokens,
            self.win_rate * 100.0
        ));

        match self.average_latency {
            Some(latency) => lines.push(format!("  average_latency={latency:.2} ticks")),
            None => lines.push("  average_latency=n/a".to_string()),
        }

        if !self.per_kind.is_empty() {
            lines.push("  token_kinds:".to_string());
            for (kind, metrics) in &self.per_kind {
                let avg_latency = metrics
                    .average_latency
                    .map(|latency| format!("{latency:.2} ticks"))
                    .unwrap_or_else(|| "n/a".to_string());
                lines.push(format!(
                    "    {}: scheduled={} started={} completed={} failed={} win_rate={:.1}% avg_latency={avg_latency}",
                    kind.as_str(),
                    metrics.scheduled,
                    metrics.started,
                    metrics.completed,
                    metrics.failed,
                    metrics.win_rate() * 100.0
                ));
            }
        }

        if !self.capability_usage.is_empty() {
            lines.push("  capabilities:".to_string());
            for (name, usage) in &self.capability_usage {
                lines.push(format!(
                    "    {} (handle {}): invocations={} tokens_consumed={}",
                    name, usage.handle, usage.invocations, usage.tokens_consumed
                ));
                if let Some(invocations) = usage.last_reported_invocations {
                    lines.push(format!("      last_reported_invocations={invocations}"));
                }
                if let Some(tokens) = usage.last_reported_tokens {
                    lines.push(format!("      last_reported_tokens={tokens}"));
                }
            }
        }

        if let Some(usage) = self.resource_usage {
            lines.push("  resources:".to_string());
            lines.push(format!(
                "    cpu: consumed={} limit={}",
                usage.cpu.consumed,
                usage
                    .cpu
                    .limit
                    .map(|value| value.to_string())
                    .unwrap_or_else(|| "unlimited".into())
            ));
            lines.push(format!(
                "    memory: consumed={} limit={}",
                usage.memory.consumed,
                usage
                    .memory
                    .limit
                    .map(|value| value.to_string())
                    .unwrap_or_else(|| "unlimited".into())
            ));
            lines.push(format!(
                "    network: consumed={} limit={}",
                usage.network.consumed,
                usage
                    .network
                    .limit
                    .map(|value| value.to_string())
                    .unwrap_or_else(|| "unlimited".into())
            ));
            lines.push(format!(
                "    tokens: consumed={} limit={}",
                usage.tokens.consumed,
                usage
                    .tokens
                    .limit
                    .map(|value| value.to_string())
                    .unwrap_or_else(|| "unlimited".into())
            ));
        }

        lines.join("\n")
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenRunMetrics {
    pub token_id: TokenId,
    pub kind: Option<TokenKind>,
    pub goal: Option<String>,
    pub scheduled_at: Option<u64>,
    pub started_at: Option<u64>,
    pub completed_at: Option<u64>,
    pub latency: Option<u64>,
    pub succeeded: bool,
    pub failed: bool,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TokenKindMetrics {
    pub scheduled: usize,
    pub started: usize,
    pub completed: usize,
    pub failed: usize,
    pub average_latency: Option<f64>,
}

impl TokenKindMetrics {
    pub fn win_rate(&self) -> f64 {
        let attempts = self.completed + self.failed;
        if attempts == 0 {
            0.0
        } else {
            self.completed as f64 / attempts as f64
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CapabilityUsageMetrics {
    pub handle: usize,
    pub name: String,
    pub invocations: u64,
    pub tokens_consumed: u64,
    pub last_reported_invocations: Option<u64>,
    pub last_reported_tokens: Option<u64>,
}

#[derive(Default)]
struct MetricsBuilder {
    tokens: BTreeMap<TokenId, TokenRunAccumulator>,
    per_kind: BTreeMap<TokenKind, TokenKindAccumulator>,
    capability_usage: BTreeMap<String, CapabilityUsageMetrics>,
    latency_sum: u128,
    latency_count: u128,
    resource_usage: Option<ResourceUsageSnapshot>,
}

impl MetricsBuilder {
    fn collect(mut self, events: &[KernelEvent]) -> ExecutionMetrics {
        for event in events {
            let token_id = event.token_id;
            let accumulator = self
                .tokens
                .entry(token_id)
                .or_insert_with(|| TokenRunAccumulator::new(token_id));

            match &event.kind {
                EventKind::Scheduled => {
                    accumulator.scheduled_at = Some(event.timestamp);
                    if let Some(kind) = parse_kind(&event.detail) {
                        accumulator.kind = Some(kind);
                        self.per_kind.entry(kind).or_default().scheduled += 1;
                    }
                    if let Some(goal) = event.detail.get("goal").and_then(|v| v.as_str()) {
                        accumulator.goal = Some(goal.to_owned());
                    }
                }
                EventKind::Started => {
                    accumulator.started_at = Some(event.timestamp);
                    if let Some(kind) = accumulator.kind {
                        self.per_kind.entry(kind).or_default().started += 1;
                    }
                    if let Some((name, usage)) = parse_capability_usage(&event.detail) {
                        let entry =
                            self.capability_usage
                                .entry(name.clone())
                                .or_insert_with(|| CapabilityUsageMetrics {
                                    handle: usage.handle,
                                    name,
                                    ..Default::default()
                                });
                        entry.invocations += 1;
                        entry.tokens_consumed += usage.tokens_consumed;
                        entry.last_reported_invocations = usage.last_reported_invocations;
                        entry.last_reported_tokens = usage.last_reported_tokens;
                    }
                    if let Some(usage) = parse_resource_usage(&event.detail) {
                        self.resource_usage = Some(usage);
                    }
                }
                EventKind::Completed => {
                    if !accumulator.succeeded {
                        accumulator.succeeded = true;
                        accumulator.completed_at = Some(event.timestamp);
                        if let Some(start) = accumulator.started_at {
                            let latency = event.timestamp.saturating_sub(start);
                            accumulator.latency = Some(latency);
                            self.latency_sum += latency as u128;
                            self.latency_count += 1;
                            if let Some(kind) = accumulator.kind {
                                let kind_entry = self.per_kind.entry(kind).or_default();
                                kind_entry.completed += 1;
                                kind_entry.latency_sum += latency as u128;
                                kind_entry.latency_count += 1;
                            }
                        } else if let Some(kind) = accumulator.kind {
                            self.per_kind.entry(kind).or_default().completed += 1;
                        }
                    }
                }
                EventKind::CapabilityViolation
                | EventKind::BudgetViolation
                | EventKind::ResourceViolation => {
                    if !accumulator.failed {
                        accumulator.failed = true;
                        if let Some(kind) = accumulator.kind {
                            self.per_kind.entry(kind).or_default().failed += 1;
                        }
                    }
                }
                _ => {}
            }
        }

        let token_runs = self
            .tokens
            .into_iter()
            .map(|(id, accumulator)| (id, accumulator.into_metrics()))
            .collect::<BTreeMap<_, _>>();

        let mut per_kind = BTreeMap::new();
        for (kind, accumulator) in self.per_kind {
            let average_latency = if accumulator.latency_count > 0 {
                Some(accumulator.latency_sum as f64 / accumulator.latency_count as f64)
            } else {
                None
            };
            per_kind.insert(
                kind,
                TokenKindMetrics {
                    scheduled: accumulator.scheduled,
                    started: accumulator.started,
                    completed: accumulator.completed,
                    failed: accumulator.failed,
                    average_latency,
                },
            );
        }

        let total_tokens = token_runs.len();
        let completed_tokens = token_runs.values().filter(|run| run.succeeded).count();
        let failed_tokens = token_runs.values().filter(|run| run.failed).count();
        let attempts = completed_tokens + failed_tokens;
        let win_rate = if attempts == 0 {
            0.0
        } else {
            completed_tokens as f64 / attempts as f64
        };

        let average_latency = if self.latency_count > 0 {
            Some(self.latency_sum as f64 / self.latency_count as f64)
        } else {
            None
        };

        ExecutionMetrics {
            total_tokens,
            completed_tokens,
            failed_tokens,
            win_rate,
            average_latency,
            token_runs,
            per_kind,
            capability_usage: self.capability_usage,
            resource_usage: self.resource_usage,
        }
    }
}

struct TokenRunAccumulator {
    token_id: TokenId,
    kind: Option<TokenKind>,
    goal: Option<String>,
    scheduled_at: Option<u64>,
    started_at: Option<u64>,
    completed_at: Option<u64>,
    latency: Option<u64>,
    succeeded: bool,
    failed: bool,
}

impl TokenRunAccumulator {
    fn new(token_id: TokenId) -> Self {
        Self {
            token_id,
            kind: None,
            goal: None,
            scheduled_at: None,
            started_at: None,
            completed_at: None,
            latency: None,
            succeeded: false,
            failed: false,
        }
    }

    fn into_metrics(self) -> TokenRunMetrics {
        TokenRunMetrics {
            token_id: self.token_id,
            kind: self.kind,
            goal: self.goal,
            scheduled_at: self.scheduled_at,
            started_at: self.started_at,
            completed_at: self.completed_at,
            latency: self.latency,
            succeeded: self.succeeded,
            failed: self.failed,
        }
    }
}

#[derive(Default)]
struct TokenKindAccumulator {
    scheduled: usize,
    started: usize,
    completed: usize,
    failed: usize,
    latency_sum: u128,
    latency_count: u128,
}

struct CapabilityUsageDetail {
    handle: usize,
    tokens_consumed: u64,
    last_reported_invocations: Option<u64>,
    last_reported_tokens: Option<u64>,
}

fn parse_kind(detail: &serde_json::Value) -> Option<TokenKind> {
    if let Some(value) = detail.get("kind") {
        if let Some(string) = value.as_str() {
            return TokenKind::from_str(string);
        }
        return serde_json::from_value(value.clone()).ok();
    }
    None
}

fn parse_capability_usage(detail: &serde_json::Value) -> Option<(String, CapabilityUsageDetail)> {
    let name = detail.get("capability_name").and_then(|v| v.as_str());
    let handle = detail
        .get("capability_handle")
        .and_then(|v| v.as_u64())
        .map(|value| value as usize)?;

    let tokens_consumed = detail
        .get("tokens_consumed")
        .and_then(|v| v.as_u64())
        .unwrap_or_default();

    let last_reported_invocations = detail.get("invocations").and_then(|v| v.as_u64());
    let last_reported_tokens = detail.get("tokens_total").and_then(|v| v.as_u64());

    let name = name
        .map(|value| value.to_owned())
        .unwrap_or_else(|| format!("capability#{handle}"));

    Some((
        name,
        CapabilityUsageDetail {
            handle,
            tokens_consumed,
            last_reported_invocations,
            last_reported_tokens,
        },
    ))
}

fn parse_resource_usage(detail: &serde_json::Value) -> Option<ResourceUsageSnapshot> {
    let resources = detail.get("resources")?;
    let usage = resources.get("usage")?;
    serde_json::from_value(usage.clone()).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aggregates_metrics_from_events() {
        let events = vec![
            KernelEvent {
                token_id: TokenId::new(1),
                kind: EventKind::Scheduled,
                detail: serde_json::json!({"kind": "Reason", "goal": "plan"}),
                timestamp: 1,
            },
            KernelEvent {
                token_id: TokenId::new(1),
                kind: EventKind::Started,
                detail: serde_json::json!({
                    "resources": {
                        "requested": {
                            "cpu_time": 3,
                            "memory_bytes": 0,
                            "network_bytes": 0,
                            "tokens": 5
                        },
                        "usage": {
                            "cpu": {"limit": null, "consumed": 3},
                            "memory": {"limit": null, "consumed": 0},
                            "network": {"limit": null, "consumed": 0},
                            "tokens": {"limit": 10, "consumed": 5}
                        }
                    },
                    "capability_handle": 0,
                    "capability_name": "reason",
                    "invocations": 1,
                    "tokens_total": 5,
                    "tokens_consumed": 5
                }),
                timestamp: 2,
            },
            KernelEvent {
                token_id: TokenId::new(1),
                kind: EventKind::Completed,
                detail: serde_json::Value::Null,
                timestamp: 5,
            },
            KernelEvent {
                token_id: TokenId::new(2),
                kind: EventKind::Scheduled,
                detail: serde_json::json!({"kind": "Act"}),
                timestamp: 3,
            },
            KernelEvent {
                token_id: TokenId::new(2),
                kind: EventKind::CapabilityViolation,
                detail: serde_json::json!({"error": "token_budget_exceeded"}),
                timestamp: 4,
            },
        ];

        let metrics = ExecutionMetrics::from_events(&events);
        assert_eq!(metrics.total_tokens, 2);
        assert_eq!(metrics.completed_tokens, 1);
        assert_eq!(metrics.failed_tokens, 1);
        assert!(metrics.win_rate > 0.0 && metrics.win_rate < 1.0);
        assert_eq!(metrics.capability_usage.len(), 1);
        let usage = metrics.capability_usage.get("reason").unwrap();
        assert_eq!(usage.invocations, 1);
        assert_eq!(usage.tokens_consumed, 5);
        let resource_usage = metrics.resource_usage.expect("resource usage");
        assert_eq!(resource_usage.tokens.consumed, 5);
        assert_eq!(resource_usage.tokens.limit, Some(10));
        let reason_metrics = metrics
            .per_kind
            .get(&TokenKind::Reason)
            .expect("reason metrics");
        assert_eq!(reason_metrics.completed, 1);
        assert_eq!(reason_metrics.failed, 0);
        assert!(reason_metrics.average_latency.is_some());
    }
}
