use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CapabilityHandle(pub(crate) usize);

impl CapabilityHandle {
    pub fn raw(&self) -> usize {
        self.0
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CapabilityLimits {
    pub max_invocations: Option<u64>,
    pub max_tokens: Option<u64>,
}

impl CapabilityLimits {
    pub fn unlimited() -> Self {
        Self {
            max_invocations: None,
            max_tokens: None,
        }
    }
}

impl Default for CapabilityLimits {
    fn default() -> Self {
        Self::unlimited()
    }
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct CapabilityUsage {
    pub invocations: u64,
    pub tokens: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CapabilityRecord {
    name: String,
    limits: CapabilityLimits,
    usage: CapabilityUsage,
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum CapabilityError {
    #[error("capability not found")]
    NotFound,
    #[error("capability invocation budget exhausted")]
    InvocationBudgetExceeded,
    #[error("capability token budget exhausted")]
    TokenBudgetExceeded,
}

#[derive(Debug, Default)]
pub struct CapabilityRegistry {
    inner: HashMap<CapabilityHandle, CapabilityRecord>,
    by_name: HashMap<String, CapabilityHandle>,
    next_handle: usize,
}

impl CapabilityRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(
        &mut self,
        name: impl Into<String>,
        limits: CapabilityLimits,
    ) -> CapabilityHandle {
        let name = name.into();

        if let Some(handle) = self.by_name.get(&name).copied() {
            if let Some(record) = self.inner.get_mut(&handle) {
                record.limits = limits;
            }
            return handle;
        }

        let handle = CapabilityHandle(self.next_handle);
        self.next_handle += 1;
        let record = CapabilityRecord {
            name: name.clone(),
            limits,
            usage: CapabilityUsage::default(),
        };
        self.by_name.insert(name, handle);
        self.inner.insert(handle, record);
        handle
    }

    pub fn info(&self, handle: CapabilityHandle) -> Option<CapabilityInfo<'_>> {
        self.inner.get(&handle).map(|record| CapabilityInfo {
            handle,
            name: &record.name,
            limits: &record.limits,
            usage: &record.usage,
        })
    }

    pub fn handle_by_name(&self, name: &str) -> Option<CapabilityHandle> {
        self.by_name.get(name).copied()
    }

    pub fn consume(
        &mut self,
        handle: CapabilityHandle,
        tokens: u64,
    ) -> Result<CapabilityUsage, CapabilityError> {
        let record = self
            .inner
            .get_mut(&handle)
            .ok_or(CapabilityError::NotFound)?;

        if let Some(max_invocations) = record.limits.max_invocations {
            if record.usage.invocations >= max_invocations {
                return Err(CapabilityError::InvocationBudgetExceeded);
            }
        }

        if let Some(max_tokens) = record.limits.max_tokens {
            if record.usage.tokens + tokens > max_tokens {
                return Err(CapabilityError::TokenBudgetExceeded);
            }
        }

        record.usage.invocations += 1;
        record.usage.tokens += tokens;
        Ok(record.usage.clone())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CapabilityInfo<'a> {
    pub handle: CapabilityHandle,
    pub name: &'a str,
    pub limits: &'a CapabilityLimits,
    pub usage: &'a CapabilityUsage,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registering_capabilities_assigns_unique_handles() {
        let mut registry = CapabilityRegistry::new();
        let first = registry.register("net", CapabilityLimits::unlimited());
        let second = registry.register("storage", CapabilityLimits::unlimited());
        assert_ne!(first, second);
        assert!(registry.info(first).is_some());
        assert!(registry.info(second).is_some());
    }

    #[test]
    fn enforcing_invocation_budget() {
        let mut registry = CapabilityRegistry::new();
        let handle = registry.register(
            "limited",
            CapabilityLimits {
                max_invocations: Some(1),
                max_tokens: None,
            },
        );

        assert!(registry.consume(handle, 0).is_ok());
        let err = registry.consume(handle, 0).unwrap_err();
        assert_eq!(err, CapabilityError::InvocationBudgetExceeded);
    }

    #[test]
    fn enforcing_token_budget() {
        let mut registry = CapabilityRegistry::new();
        let handle = registry.register(
            "limited",
            CapabilityLimits {
                max_invocations: None,
                max_tokens: Some(2),
            },
        );

        assert!(registry.consume(handle, 1).is_ok());
        let err = registry.consume(handle, 2).unwrap_err();
        assert_eq!(err, CapabilityError::TokenBudgetExceeded);
    }

    #[test]
    fn lookup_by_name_reuses_existing_handles() {
        let mut registry = CapabilityRegistry::new();
        let first = registry.register("tool", CapabilityLimits::default());
        assert_eq!(registry.handle_by_name("tool"), Some(first));

        assert!(registry.consume(first, 1).is_ok());

        let second = registry.register(
            "tool",
            CapabilityLimits {
                max_invocations: Some(5),
                max_tokens: Some(10),
            },
        );

        assert_eq!(first, second);
        let info = registry.info(second).expect("info available");
        assert_eq!(info.usage.tokens, 1);
        assert_eq!(info.limits.max_invocations, Some(5));
        assert_eq!(info.limits.max_tokens, Some(10));
    }
}
