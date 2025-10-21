use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};

use serde::{Deserialize, Serialize};

use crate::capability::CapabilityHandle;
use crate::resource::ResourceRequest;
use crate::token_signing::TokenSignature;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, PartialOrd, Ord)]
pub struct TokenId(u64);

impl TokenId {
    pub fn new(id: u64) -> Self {
        Self(id)
    }

    pub fn raw(&self) -> u64 {
        self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, PartialOrd, Ord)]
pub enum TokenKind {
    Perception,
    Reason,
    Plan,
    Act,
    Reflect,
}

impl TokenKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            TokenKind::Perception => "Perception",
            TokenKind::Reason => "Reason",
            TokenKind::Plan => "Plan",
            TokenKind::Act => "Act",
            TokenKind::Reflect => "Reflect",
        }
    }

    pub fn from_str(value: &str) -> Option<Self> {
        match value {
            "Perception" => Some(TokenKind::Perception),
            "Reason" => Some(TokenKind::Reason),
            "Plan" => Some(TokenKind::Plan),
            "Act" => Some(TokenKind::Act),
            "Reflect" => Some(TokenKind::Reflect),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct TokenBudget {
    pub limit: u64,
    pub consumed: u64,
}

impl TokenBudget {
    pub fn new(limit: u64) -> Self {
        Self { limit, consumed: 0 }
    }

    pub fn remaining(&self) -> u64 {
        self.limit.saturating_sub(self.consumed)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KernelToken {
    pub id: TokenId,
    pub priority: u8,
    pub cost: u64,
    pub kind: TokenKind,
    pub capability: Option<CapabilityHandle>,
    pub payload: serde_json::Value,
    pub context_hash: String,
    pub goal: Option<String>,
    pub granted_capabilities: Vec<CapabilityHandle>,
    pub dependencies: Vec<TokenId>,
    pub budget: TokenBudget,
    pub resources: ResourceRequest,
    #[serde(skip_serializing, skip_deserializing)]
    pub signature: Option<TokenSignature>,
}

impl KernelToken {
    pub fn new(id: TokenId, priority: u8, cost: u64, kind: TokenKind) -> Self {
        Self {
            id,
            priority,
            cost,
            kind,
            capability: None,
            payload: serde_json::Value::Null,
            context_hash: String::new(),
            goal: None,
            granted_capabilities: Vec::new(),
            dependencies: Vec::new(),
            budget: TokenBudget::new(cost),
            resources: ResourceRequest::default(),
            signature: None,
        }
    }

    pub fn with_capability(mut self, capability: CapabilityHandle) -> Self {
        self.capability = Some(capability);
        self
    }

    pub fn with_payload(mut self, payload: serde_json::Value) -> Self {
        self.payload = payload;
        self
    }

    pub fn with_context_hash(mut self, hash: impl Into<String>) -> Self {
        self.context_hash = hash.into();
        self
    }

    pub fn with_goal(mut self, goal: impl Into<String>) -> Self {
        self.goal = Some(goal.into());
        self
    }

    pub fn with_granted_capabilities(
        mut self,
        capabilities: impl Into<Vec<CapabilityHandle>>,
    ) -> Self {
        self.granted_capabilities = capabilities.into();
        self
    }

    pub fn with_dependencies(mut self, dependencies: impl Into<Vec<TokenId>>) -> Self {
        self.dependencies = dependencies.into();
        self
    }

    pub fn with_budget(mut self, budget: TokenBudget) -> Self {
        self.budget = budget;
        self
    }

    pub fn with_resource_request(mut self, request: ResourceRequest) -> Self {
        self.resources = request;
        self
    }

    pub fn with_signature(mut self, signature: TokenSignature) -> Self {
        self.signature = Some(signature);
        self
    }
}

#[derive(Debug, Clone)]
struct QueueEntry {
    priority: u8,
    sequence: u64,
    token: KernelToken,
}

impl Ord for QueueEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        match self.priority.cmp(&other.priority) {
            Ordering::Equal => other.sequence.cmp(&self.sequence),
            other => other,
        }
    }
}

impl PartialOrd for QueueEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for QueueEntry {
    fn eq(&self, other: &Self) -> bool {
        self.priority == other.priority
            && self.sequence == other.sequence
            && self.token == other.token
    }
}

impl Eq for QueueEntry {}

#[derive(Debug, Default)]
pub struct Scheduler {
    queue: BinaryHeap<QueueEntry>,
    sequence: u64,
    waiting: Vec<QueueEntry>,
    completed: HashSet<TokenId>,
    budgets: HashMap<TokenId, TokenBudget>,
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum SchedulerError {
    #[error("token {token_id:?} exceeded budget (attempted {attempted}, remaining {remaining})")]
    BudgetExceeded {
        token_id: TokenId,
        attempted: u64,
        remaining: u64,
    },
    #[error("token {token_id:?} has no registered budget")]
    MissingBudget { token_id: TokenId },
}

impl Scheduler {
    pub fn new() -> Self {
        Self::default()
    }

    /// Schedule a token for execution using deterministic ordering.
    ///
    /// ```
    /// use os2_kernel::scheduler::{Scheduler, KernelToken, TokenId, TokenKind};
    /// use os2_kernel::resource::ResourceRequest;
    /// use serde_json::json;
    ///
    /// let mut scheduler = Scheduler::new();
    /// let token = KernelToken::new(TokenId::new(1), 5, 1, TokenKind::Plan)
    ///     .with_payload(json!({"demo": true}))
    ///     .with_resource_request(ResourceRequest::default());
    /// scheduler.schedule(token.clone());
    /// let next = scheduler.next().expect("scheduled token available");
    /// assert_eq!(next.id.raw(), token.id.raw());
    /// ```
    pub fn schedule(&mut self, token: KernelToken) {
        let entry = QueueEntry {
            priority: token.priority,
            sequence: self.sequence,
            token,
        };
        self.sequence += 1;

        let token_id = entry.token.id;
        self.budgets
            .entry(token_id)
            .or_insert(entry.token.budget.clone());

        if self.dependencies_satisfied(&entry.token) {
            self.queue.push(entry);
        } else {
            self.waiting.push(entry);
        }
    }

    fn dependencies_satisfied(&self, token: &KernelToken) -> bool {
        token
            .dependencies
            .iter()
            .all(|dep| self.completed.contains(dep))
    }

    fn refresh_waiting(&mut self) {
        let waiting = std::mem::take(&mut self.waiting);
        let mut remaining = Vec::new();
        for entry in waiting {
            if self.dependencies_satisfied(&entry.token) {
                self.queue.push(entry);
            } else {
                remaining.push(entry);
            }
        }
        self.waiting = remaining;
    }

    pub fn start(&mut self, token: TokenId, cost: u64) -> Result<(), SchedulerError> {
        let budget = self
            .budgets
            .get_mut(&token)
            .ok_or(SchedulerError::MissingBudget { token_id: token })?;

        if budget.remaining() < cost {
            return Err(SchedulerError::BudgetExceeded {
                token_id: token,
                attempted: cost,
                remaining: budget.remaining(),
            });
        }

        budget.consumed += cost;
        Ok(())
    }

    pub fn next(&mut self) -> Option<KernelToken> {
        if self.queue.is_empty() {
            self.refresh_waiting();
        }

        while let Some(entry) = self.queue.pop() {
            if self.dependencies_satisfied(&entry.token) {
                return Some(entry.token);
            }
            self.waiting.push(entry);
        }

        None
    }

    pub fn is_empty(&self) -> bool {
        self.queue.is_empty() && self.waiting.is_empty()
    }

    pub fn len(&self) -> usize {
        self.queue.len() + self.waiting.len()
    }

    pub fn complete(&mut self, token_id: TokenId) {
        self.completed.insert(token_id);
        self.refresh_waiting();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scheduler_orders_by_priority_and_fifo() {
        let mut scheduler = Scheduler::new();
        scheduler.schedule(KernelToken::new(TokenId::new(1), 10, 1, TokenKind::Reason));
        scheduler.schedule(KernelToken::new(TokenId::new(2), 20, 1, TokenKind::Reason));
        scheduler.schedule(KernelToken::new(TokenId::new(3), 20, 1, TokenKind::Reason));

        let first = scheduler.next().unwrap();
        scheduler.complete(first.id);
        let second = scheduler.next().unwrap();
        scheduler.complete(second.id);
        let third = scheduler.next().unwrap();

        assert_eq!(first.id.raw(), 2);
        assert_eq!(second.id.raw(), 3);
        assert_eq!(third.id.raw(), 1);
    }

    #[test]
    fn scheduler_blocks_on_dependencies() {
        let mut scheduler = Scheduler::new();
        let root = KernelToken::new(TokenId::new(1), 10, 1, TokenKind::Perception);
        let dependent = KernelToken::new(TokenId::new(2), 20, 1, TokenKind::Reason)
            .with_dependencies(vec![TokenId::new(1)]);

        scheduler.schedule(dependent.clone());
        scheduler.schedule(root.clone());

        let first = scheduler.next().unwrap();
        assert_eq!(first.id, root.id);
        scheduler.complete(first.id);

        let second = scheduler.next().unwrap();
        assert_eq!(second.id, dependent.id);
    }

    #[test]
    fn scheduler_enforces_budgets() {
        let mut scheduler = Scheduler::new();
        let token = KernelToken::new(TokenId::new(1), 10, 5, TokenKind::Act)
            .with_budget(TokenBudget::new(3));
        scheduler.schedule(token);

        let next = scheduler.next().unwrap();
        let err = scheduler.start(next.id, next.cost).unwrap_err();
        assert!(matches!(err, SchedulerError::BudgetExceeded { .. }));
    }
}
