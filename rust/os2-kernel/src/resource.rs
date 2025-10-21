use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ResourceType {
    Cpu,
    Memory,
    Network,
    Tokens,
}

impl ResourceType {
    pub fn as_str(&self) -> &'static str {
        match self {
            ResourceType::Cpu => "cpu",
            ResourceType::Memory => "memory",
            ResourceType::Network => "network",
            ResourceType::Tokens => "tokens",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct ResourceRequest {
    pub cpu_time: u64,
    pub memory_bytes: u64,
    pub network_bytes: u64,
    pub tokens: u64,
}

impl ResourceRequest {
    pub fn cpu(cpu_time: u64) -> Self {
        Self {
            cpu_time,
            ..Self::default()
        }
    }

    pub fn memory(memory_bytes: u64) -> Self {
        Self {
            memory_bytes,
            ..Self::default()
        }
    }

    pub fn network(network_bytes: u64) -> Self {
        Self {
            network_bytes,
            ..Self::default()
        }
    }

    pub fn tokens(tokens: u64) -> Self {
        Self {
            tokens,
            ..Self::default()
        }
    }

    pub fn combine(mut self, other: Self) -> Self {
        self.cpu_time = self.cpu_time.saturating_add(other.cpu_time);
        self.memory_bytes = self.memory_bytes.saturating_add(other.memory_bytes);
        self.network_bytes = self.network_bytes.saturating_add(other.network_bytes);
        self.tokens = self.tokens.saturating_add(other.tokens);
        self
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResourceQuota {
    pub limit: Option<u64>,
    pub consumed: u64,
}

impl ResourceQuota {
    pub fn unlimited() -> Self {
        Self {
            limit: None,
            consumed: 0,
        }
    }

    fn can_consume(&self, amount: u64, resource: ResourceType) -> Result<(), ResourceError> {
        if amount == 0 {
            return Ok(());
        }

        if let Some(limit) = self.limit {
            let projected = self.consumed.saturating_add(amount);
            if projected > limit {
                let remaining = limit.saturating_sub(self.consumed);
                return Err(ResourceError::BudgetExceeded {
                    resource,
                    attempted: amount,
                    remaining,
                });
            }
        }

        Ok(())
    }

    fn apply(&mut self, amount: u64) {
        if amount == 0 {
            return;
        }
        self.consumed = self.consumed.saturating_add(amount);
    }
}

impl Default for ResourceQuota {
    fn default() -> Self {
        Self::unlimited()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct ResourceUsageSnapshot {
    pub cpu: ResourceQuota,
    pub memory: ResourceQuota,
    pub network: ResourceQuota,
    pub tokens: ResourceQuota,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResourceGovernor {
    cpu: ResourceQuota,
    memory: ResourceQuota,
    network: ResourceQuota,
    tokens: ResourceQuota,
}

impl Default for ResourceGovernor {
    fn default() -> Self {
        Self {
            cpu: ResourceQuota::default(),
            memory: ResourceQuota::default(),
            network: ResourceQuota::default(),
            tokens: ResourceQuota::default(),
        }
    }
}

impl ResourceGovernor {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn set_limit(&mut self, resource: ResourceType, limit: Option<u64>) {
        let quota = self.quota_mut(resource);
        quota.limit = limit;
        if let Some(limit) = quota.limit {
            quota.consumed = quota.consumed.min(limit);
        }
    }

    pub fn reset_usage(&mut self) {
        self.cpu.consumed = 0;
        self.memory.consumed = 0;
        self.network.consumed = 0;
        self.tokens.consumed = 0;
    }

    pub fn usage(&self) -> ResourceUsageSnapshot {
        ResourceUsageSnapshot {
            cpu: self.cpu,
            memory: self.memory,
            network: self.network,
            tokens: self.tokens,
        }
    }

    pub fn consume(
        &mut self,
        request: &ResourceRequest,
    ) -> Result<ResourceUsageSnapshot, ResourceError> {
        self.cpu.can_consume(request.cpu_time, ResourceType::Cpu)?;
        self.memory
            .can_consume(request.memory_bytes, ResourceType::Memory)?;
        self.network
            .can_consume(request.network_bytes, ResourceType::Network)?;
        self.tokens
            .can_consume(request.tokens, ResourceType::Tokens)?;

        self.cpu.apply(request.cpu_time);
        self.memory.apply(request.memory_bytes);
        self.network.apply(request.network_bytes);
        self.tokens.apply(request.tokens);

        Ok(self.usage())
    }

    fn quota_mut(&mut self, resource: ResourceType) -> &mut ResourceQuota {
        match resource {
            ResourceType::Cpu => &mut self.cpu,
            ResourceType::Memory => &mut self.memory,
            ResourceType::Network => &mut self.network,
            ResourceType::Tokens => &mut self.tokens,
        }
    }
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum ResourceError {
    #[error("{resource:?} budget exceeded (attempted {attempted}, remaining {remaining})")]
    BudgetExceeded {
        resource: ResourceType,
        attempted: u64,
        remaining: u64,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn consuming_within_limits_updates_usage() {
        let mut governor = ResourceGovernor::new();
        governor.set_limit(ResourceType::Cpu, Some(10));
        governor.set_limit(ResourceType::Tokens, Some(20));

        let request = ResourceRequest {
            cpu_time: 4,
            memory_bytes: 0,
            network_bytes: 0,
            tokens: 5,
        };

        let usage = governor.consume(&request).expect("within budget");
        assert_eq!(usage.cpu.consumed, 4);
        assert_eq!(usage.cpu.limit, Some(10));
        assert_eq!(usage.tokens.consumed, 5);
        assert_eq!(usage.tokens.limit, Some(20));
    }

    #[test]
    fn exceeding_limits_errors() {
        let mut governor = ResourceGovernor::new();
        governor.set_limit(ResourceType::Memory, Some(8));

        let request = ResourceRequest {
            cpu_time: 0,
            memory_bytes: 12,
            network_bytes: 0,
            tokens: 0,
        };

        let err = governor.consume(&request).unwrap_err();
        assert_eq!(
            err,
            ResourceError::BudgetExceeded {
                resource: ResourceType::Memory,
                attempted: 12,
                remaining: 8,
            }
        );
    }
}
