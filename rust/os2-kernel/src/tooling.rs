use std::collections::HashMap;

use anyhow::{Context, Result, anyhow};

use crate::capability::{CapabilityHandle, CapabilityLimits};
use crate::kernel::Kernel;

pub trait ToolAdapter: Send + Sync {
    fn name(&self) -> &str;
    fn capability_limits(&self) -> CapabilityLimits;
    fn cost(&self, input: &serde_json::Value) -> u64 {
        let _ = input;
        1
    }
    fn invoke(&self, input: &serde_json::Value) -> Result<serde_json::Value>;
}

pub struct ToolCatalog {
    entries: HashMap<String, ToolEntry>,
}

struct ToolEntry {
    capability: CapabilityHandle,
    adapter: Box<dyn ToolAdapter>,
}

impl ToolCatalog {
    pub fn from_adapters(kernel: &mut Kernel, adapters: Vec<Box<dyn ToolAdapter>>) -> Result<Self> {
        let mut entries = HashMap::new();
        for adapter in adapters {
            let name = adapter.name().to_string();
            if entries.contains_key(&name) {
                return Err(anyhow!("tool `{}` registered twice", name));
            }
            let limits = adapter.capability_limits();
            let capability = kernel.register_capability(name.clone(), limits);
            entries.insert(
                name,
                ToolEntry {
                    capability,
                    adapter,
                },
            );
        }

        Ok(Self { entries })
    }

    pub fn invoke(
        &self,
        kernel: &mut Kernel,
        name: &str,
        payload: &serde_json::Value,
    ) -> Result<serde_json::Value> {
        let entry = self
            .entries
            .get(name)
            .ok_or_else(|| anyhow!("tool `{}` not registered", name))?;

        let cost = entry.adapter.cost(payload);
        kernel
            .consume_capability(entry.capability, cost)
            .with_context(|| format!("consuming capability for `{name}`"))?;

        entry.adapter.invoke(payload)
    }
}

#[derive(Debug, Clone)]
pub struct ToolInvocationResult {
    pub tool: String,
    pub output: serde_json::Value,
}
