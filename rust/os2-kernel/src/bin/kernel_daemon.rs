use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs::File;
use std::io::{BufReader, BufWriter, Write};
use std::path::PathBuf;
use std::{env, process};

use os2_kernel::{
    CapabilityLimits, EventKind, Kernel, KernelEvent, KernelToken, ResourceRequest, ResourceType,
    TokenExecutor, TokenKind,
};
use os2_kernel::{ExecutionContext, ExecutionStatus};
use serde::Deserialize;
use serde_json::Value;

fn main() {
    if let Err(err) = run() {
        eprintln!("error: {err}");
        process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1);
    let config_path = args.next().ok_or_else(
        || "missing configuration path. usage: kernel_daemon <config> [events.jsonl]",
    )?;
    let output_path = args.next().map(PathBuf::from);

    let file = File::open(&config_path)?;
    let reader = BufReader::new(file);
    let config: DaemonConfig = serde_json::from_reader(reader)?;

    let mut kernel = Kernel::new(config.seed);
    config.resource_limits.apply(&mut kernel);
    let mut handles: HashMap<String, os2_kernel::CapabilityHandle> = HashMap::new();

    for capability in config.capabilities {
        let handle = kernel.register_capability(capability.name.clone(), capability.limits);
        handles.insert(capability.name, handle);
    }

    for token_spec in config.tokens {
        let mut token =
            kernel.allocate_token(token_spec.priority, token_spec.cost, token_spec.kind);
        if let Some(capability_name) = token_spec.capability.as_ref() {
            let handle = handles
                .get(capability_name)
                .copied()
                .ok_or_else(|| format!("unknown capability '{capability_name}'"))?;
            token = token.with_capability(handle);
        }
        token = token
            .with_payload(token_spec.payload)
            .with_resource_request(token_spec.resources.to_request());
        kernel.submit_token(token);
    }

    let mut executor = ScriptedExecutor::default();
    kernel.process_until_idle(&mut executor);

    let events = kernel.drain_events();

    if let Some(path) = output_path {
        write_events(&events, path)?;
    }

    print_summary(&events);

    Ok(())
}

fn write_events(events: &[KernelEvent], path: PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    let file = File::create(&path)?;
    let mut writer = BufWriter::new(file);

    for event in events {
        let line = serde_json::to_string(&event.to_json())?;
        writeln!(writer, "{}", line)?;
    }

    writer.flush()?;
    Ok(())
}

fn print_summary(events: &[KernelEvent]) {
    if events.is_empty() {
        println!("Kernel daemon processed no events.");
        return;
    }

    println!("Kernel daemon timeline:");
    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    let mut tokens: BTreeSet<u64> = BTreeSet::new();

    for event in events {
        tokens.insert(event.token_id.raw());
        let detail = detail_to_string(&event.detail);
        match &event.kind {
            EventKind::Scheduled => println!(
                "- token {} scheduled (detail: {})",
                event.token_id.raw(),
                detail
            ),
            EventKind::Started => println!(
                "- token {} started (detail: {})",
                event.token_id.raw(),
                detail
            ),
            EventKind::Completed => println!(
                "- token {} completed (detail: {})",
                event.token_id.raw(),
                detail
            ),
            EventKind::Yielded => println!(
                "- token {} yielded (detail: {})",
                event.token_id.raw(),
                detail
            ),
            EventKind::CapabilityViolation => println!(
                "- token {} violated capability limits (detail: {})",
                event.token_id.raw(),
                detail
            ),
            EventKind::BudgetViolation => println!(
                "- token {} exhausted per-token budget (detail: {})",
                event.token_id.raw(),
                detail
            ),
            EventKind::ResourceViolation => println!(
                "- token {} violated resource governors (detail: {})",
                event.token_id.raw(),
                detail
            ),
            EventKind::SecurityViolation => println!(
                "- token {} triggered security violation (detail: {})",
                event.token_id.raw(),
                detail
            ),
            EventKind::SnapshotCreated => println!(
                "- token {} created snapshot (detail: {})",
                event.token_id.raw(),
                detail
            ),
            EventKind::SnapshotRollbackStarted => println!(
                "- token {} started snapshot rollback (detail: {})",
                event.token_id.raw(),
                detail
            ),
            EventKind::SnapshotRollbackCommitted => println!(
                "- token {} committed snapshot rollback (detail: {})",
                event.token_id.raw(),
                detail
            ),
            EventKind::SnapshotRollbackFailed => println!(
                "- token {} failed snapshot rollback (detail: {})",
                event.token_id.raw(),
                detail
            ),
            EventKind::SnapshotIntegrityVerified => println!(
                "- token {} verified snapshot ledger (detail: {})",
                event.token_id.raw(),
                detail
            ),
            EventKind::SnapshotIntegrityViolation => println!(
                "- token {} detected snapshot ledger violation (detail: {})",
                event.token_id.raw(),
                detail
            ),
            EventKind::MemoryRead => println!(
                "- token {} read semantic memory (detail: {})",
                event.token_id.raw(),
                detail
            ),
            EventKind::MemoryWrite => println!(
                "- token {} wrote semantic memory (detail: {})",
                event.token_id.raw(),
                detail
            ),
            EventKind::Checkpoint => println!(
                "- token {} checkpoint recorded (detail: {})",
                event.token_id.raw(),
                detail
            ),
            EventKind::PolicyAlert => println!(
                "- token {} policy alert (detail: {})",
                event.token_id.raw(),
                detail
            ),
            EventKind::IoQueued => println!(
                "- token {} queued I/O (detail: {})",
                event.token_id.raw(),
                detail
            ),
            EventKind::IoCompleted => println!(
                "- token {} completed I/O (detail: {})",
                event.token_id.raw(),
                detail
            ),
            EventKind::Custom(label) => println!(
                "- token {} custom:{} (detail: {})",
                event.token_id.raw(),
                label,
                detail
            ),
        }

        let key = match &event.kind {
            EventKind::Custom(label) => format!("custom({label})"),
            other => other.as_str().to_string(),
        };
        *counts.entry(key).or_default() += 1;
    }

    println!();
    println!("Kernel daemon summary:");
    println!("  unique tokens processed: {}", tokens.len());
    println!("  total events: {}", events.len());

    for (kind, count) in counts {
        println!("    {kind}: {count}");
    }
}

fn detail_to_string(value: &Value) -> String {
    if value.is_null() {
        "null".to_string()
    } else {
        serde_json::to_string(value).unwrap_or_else(|_| "<unserializable>".to_string())
    }
}

#[derive(Debug, Deserialize)]
struct DaemonConfig {
    #[serde(default)]
    seed: u64,
    #[serde(default)]
    capabilities: Vec<CapabilitySpec>,
    #[serde(default)]
    tokens: Vec<TokenSpec>,
    #[serde(default)]
    resource_limits: ResourceLimitSpec,
}

#[derive(Debug, Deserialize)]
struct CapabilitySpec {
    name: String,
    #[serde(default)]
    limits: CapabilityLimits,
}

#[derive(Debug, Deserialize)]
struct TokenSpec {
    priority: u8,
    cost: u64,
    kind: TokenKind,
    capability: Option<String>,
    #[serde(default)]
    payload: Value,
    #[serde(default)]
    resources: ResourceRequestSpec,
}

#[derive(Debug, Deserialize, Default)]
struct ResourceLimitSpec {
    #[serde(default)]
    cpu_time: Option<u64>,
    #[serde(default)]
    memory_bytes: Option<u64>,
    #[serde(default)]
    network_bytes: Option<u64>,
    #[serde(default)]
    tokens: Option<u64>,
}

impl ResourceLimitSpec {
    fn apply(&self, kernel: &mut Kernel) {
        if let Some(limit) = self.cpu_time {
            kernel.configure_resource_limit(ResourceType::Cpu, Some(limit));
        }
        if let Some(limit) = self.memory_bytes {
            kernel.configure_resource_limit(ResourceType::Memory, Some(limit));
        }
        if let Some(limit) = self.network_bytes {
            kernel.configure_resource_limit(ResourceType::Network, Some(limit));
        }
        if let Some(limit) = self.tokens {
            kernel.configure_resource_limit(ResourceType::Tokens, Some(limit));
        }
    }
}

#[derive(Debug, Deserialize, Default)]
struct ResourceRequestSpec {
    #[serde(default)]
    cpu_time: Option<u64>,
    #[serde(default)]
    memory_bytes: Option<u64>,
    #[serde(default)]
    network_bytes: Option<u64>,
    #[serde(default)]
    tokens: Option<u64>,
}

impl ResourceRequestSpec {
    fn to_request(&self) -> ResourceRequest {
        ResourceRequest {
            cpu_time: self.cpu_time.unwrap_or_default(),
            memory_bytes: self.memory_bytes.unwrap_or_default(),
            network_bytes: self.network_bytes.unwrap_or_default(),
            tokens: self.tokens.unwrap_or_default(),
        }
    }
}

#[derive(Default)]
struct ScriptedExecutor;

impl TokenExecutor for ScriptedExecutor {
    fn execute(&mut self, token: &KernelToken, ctx: &mut ExecutionContext) -> ExecutionStatus {
        if let Ok(plan) = serde_json::from_value::<ScriptedPlan>(token.payload.clone()) {
            for action in plan.actions {
                match action {
                    ScriptAction::Emit { label, detail } => {
                        ctx.emit_custom(token.id, label, detail);
                    }
                    ScriptAction::Checkpoint { label, state } => {
                        ctx.checkpoint(token.id, label, state);
                    }
                }
            }
        }

        ExecutionStatus::Completed
    }
}

#[derive(Debug, Deserialize)]
struct ScriptedPlan {
    #[serde(default)]
    actions: Vec<ScriptAction>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ScriptAction {
    Emit {
        label: String,
        #[serde(default)]
        detail: Value,
    },
    Checkpoint {
        label: String,
        #[serde(default)]
        state: Value,
    },
}
