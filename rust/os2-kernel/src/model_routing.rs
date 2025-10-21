use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum EndpointKind {
    Local,
    Remote,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ModelEndpoint {
    pub name: String,
    pub kind: EndpointKind,
    pub latency_weight: u32,
    pub cost_weight: u32,
    pub supported_capabilities: Vec<String>,
    pub max_tokens_per_minute: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RoutingRequest {
    pub context_hash: String,
    pub required_capabilities: Vec<String>,
    pub tokens: u64,
    pub priority: u8,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum RouteReason {
    Primary,
    BudgetFallback { skipped: Vec<String> },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RouteAssignment {
    pub endpoint: ModelEndpoint,
    pub reason: RouteReason,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum RouteDecision {
    Assigned(RouteAssignment),
    Rejected { reason: String },
}

pub struct DeterministicRouter {
    seed: u64,
    endpoints: Vec<ModelEndpoint>,
    budgets: HashMap<String, u64>,
}

impl DeterministicRouter {
    pub fn new(seed: u64, endpoints: Vec<ModelEndpoint>) -> Self {
        let mut budgets = HashMap::new();
        for endpoint in &endpoints {
            budgets.insert(endpoint.name.clone(), endpoint.max_tokens_per_minute);
        }
        Self {
            seed,
            endpoints,
            budgets,
        }
    }

    pub fn available_budget(&self, endpoint: &str) -> Option<u64> {
        self.budgets.get(endpoint).copied()
    }

    pub fn update_budget(&mut self, endpoint: &str, remaining: u64) {
        if let Some(slot) = self.budgets.get_mut(endpoint) {
            *slot = remaining;
        }
    }

    pub fn reset_budgets(&mut self) {
        self.budgets.clear();
        for endpoint in &self.endpoints {
            self.budgets
                .insert(endpoint.name.clone(), endpoint.max_tokens_per_minute);
        }
    }

    fn ordering_key(
        &self,
        endpoint: &ModelEndpoint,
        request: &RoutingRequest,
    ) -> (u8, u64, u32, u32) {
        let mut hasher = Sha256::new();
        hasher.update(self.seed.to_le_bytes());
        hasher.update(endpoint.name.as_bytes());
        hasher.update(request.context_hash.as_bytes());
        hasher.update([request.priority]);
        let digest = hasher.finalize();
        let mut base = [0u8; 8];
        base.copy_from_slice(&digest[..8]);
        let hashed = u64::from_le_bytes(base);
        let kind_rank = if matches!(endpoint.kind, EndpointKind::Local) {
            0
        } else {
            1
        };
        (
            kind_rank,
            hashed,
            endpoint.latency_weight,
            endpoint.cost_weight,
        )
    }

    fn supports_capabilities(endpoint: &ModelEndpoint, required: &[String]) -> bool {
        required
            .iter()
            .all(|cap| endpoint.supported_capabilities.contains(cap))
    }

    pub fn route(&mut self, request: &RoutingRequest) -> RouteDecision {
        let mut scored: Vec<(usize, (u8, u64, u32, u32))> = self
            .endpoints
            .iter()
            .enumerate()
            .filter(|(_, endpoint)| {
                Self::supports_capabilities(endpoint, &request.required_capabilities)
            })
            .map(|(idx, endpoint)| (idx, self.ordering_key(endpoint, request)))
            .collect();

        if scored.is_empty() {
            return RouteDecision::Rejected {
                reason: "no endpoint supports required capabilities".into(),
            };
        }

        scored.sort_by_key(|(_, key)| *key);

        let mut skipped = Vec::new();
        for (idx, _) in &scored {
            let endpoint = &self.endpoints[*idx];
            let available = self.available_budget(&endpoint.name).unwrap_or(0);
            if available >= request.tokens {
                let remaining = available - request.tokens;
                self.budgets.insert(endpoint.name.clone(), remaining);
                let reason = if skipped.is_empty() {
                    RouteReason::Primary
                } else {
                    RouteReason::BudgetFallback {
                        skipped: skipped.clone(),
                    }
                };
                return RouteDecision::Assigned(RouteAssignment {
                    endpoint: endpoint.clone(),
                    reason,
                });
            }
            skipped.push(endpoint.name.clone());
        }

        RouteDecision::Rejected {
            reason: format!(
                "no endpoint had sufficient budget; exhausted candidates: {}",
                skipped.join(", ")
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_endpoints() -> Vec<ModelEndpoint> {
        vec![
            ModelEndpoint {
                name: "local-primary".into(),
                kind: EndpointKind::Local,
                latency_weight: 1,
                cost_weight: 1,
                supported_capabilities: vec!["completion".into()],
                max_tokens_per_minute: 20,
            },
            ModelEndpoint {
                name: "cloud-fallback".into(),
                kind: EndpointKind::Remote,
                latency_weight: 10,
                cost_weight: 5,
                supported_capabilities: vec!["completion".into()],
                max_tokens_per_minute: 1000,
            },
        ]
    }

    fn completion_request(tokens: u64) -> RoutingRequest {
        RoutingRequest {
            context_hash: "abc123".into(),
            required_capabilities: vec!["completion".into()],
            tokens,
            priority: 10,
        }
    }

    #[test]
    fn prefers_local_until_budget_exhausted() {
        let endpoints = sample_endpoints();
        let mut router = DeterministicRouter::new(7, endpoints);

        let first = router.route(&completion_request(10));
        let second = router.route(&completion_request(10));
        let third = router.route(&completion_request(10));

        match first {
            RouteDecision::Assigned(assignment) => {
                assert_eq!(assignment.endpoint.name, "local-primary");
                assert!(matches!(assignment.reason, RouteReason::Primary));
            }
            _ => panic!("expected assignment"),
        }

        match second {
            RouteDecision::Assigned(assignment) => {
                assert_eq!(assignment.endpoint.name, "local-primary");
            }
            _ => panic!("expected assignment"),
        }

        match third {
            RouteDecision::Assigned(assignment) => {
                assert_eq!(assignment.endpoint.name, "cloud-fallback");
                match assignment.reason {
                    RouteReason::BudgetFallback { skipped } => {
                        assert_eq!(skipped, vec!["local-primary".to_string()]);
                    }
                    _ => panic!("expected budget fallback"),
                }
            }
            other => panic!("unexpected routing decision: {other:?}"),
        }
    }

    #[test]
    fn deterministic_selection_with_reset() {
        let endpoints = sample_endpoints();
        let mut router = DeterministicRouter::new(99, endpoints.clone());

        let assignment = router.route(&completion_request(5));
        router.reset_budgets();
        let assignment_after_reset = router.route(&completion_request(5));

        let first_name = match assignment {
            RouteDecision::Assigned(assignment) => assignment.endpoint.name,
            _ => panic!("expected assignment"),
        };

        let second_name = match assignment_after_reset {
            RouteDecision::Assigned(assignment) => assignment.endpoint.name,
            _ => panic!("expected assignment"),
        };

        assert_eq!(first_name, second_name);
    }

    #[test]
    fn rejects_when_capability_missing() {
        let mut router = DeterministicRouter::new(11, sample_endpoints());
        let request = RoutingRequest {
            context_hash: "zzz".into(),
            required_capabilities: vec!["embedding".into()],
            tokens: 5,
            priority: 0,
        };
        let decision = router.route(&request);
        match decision {
            RouteDecision::Rejected { reason } => {
                assert!(reason.contains("no endpoint supports"));
            }
            _ => panic!("expected rejection"),
        }
    }
}
