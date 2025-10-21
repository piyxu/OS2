use std::env;
use std::fs::File;
use std::io::{self, BufRead, BufReader, Read};
use std::path::Path;

use os2_kernel::{EventKind, ExecutionMetrics, KernelEvent, TokenId};

fn main() -> anyhow::Result<()> {
    let path = env::args().nth(1);

    let mut events = if let Some(path) = path {
        read_events_from_path(path)?
    } else {
        read_events_from_reader(io::stdin())?
    };

    if events.is_empty() {
        println!("no events to replay");
        return Ok(());
    }

    events.sort_by_key(|event| event.timestamp);

    let mut summary = ReplaySummary::default();

    for event in &events {
        println!("{}", format_event(event));
        summary.observe(event);
    }

    println!();
    println!("{}", summary.render(events.len()));

    let metrics = ExecutionMetrics::from_events(&events);
    println!("{}", metrics.render_report());

    Ok(())
}

fn read_events_from_path(path: impl AsRef<Path>) -> anyhow::Result<Vec<KernelEvent>> {
    let file = File::open(path)?;
    read_events_from_reader(file)
}

fn read_events_from_reader<R>(reader: R) -> anyhow::Result<Vec<KernelEvent>>
where
    R: Read,
{
    let reader = BufReader::new(reader);
    let mut events = Vec::new();

    for line in reader.lines() {
        let line = line?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        let value: serde_json::Value = serde_json::from_str(trimmed)?;

        match parse_event(value) {
            Ok(event) => events.push(event),
            Err(err) => eprintln!("skipping malformed event: {err}"),
        }
    }

    Ok(events)
}

fn parse_event(value: serde_json::Value) -> anyhow::Result<KernelEvent> {
    let token_id = value
        .get("token_id")
        .and_then(|v| v.as_u64())
        .ok_or_else(|| anyhow::anyhow!("missing token_id"))?;
    let timestamp = value
        .get("timestamp")
        .and_then(|v| v.as_u64())
        .ok_or_else(|| anyhow::anyhow!("missing timestamp"))?;

    let kind = value
        .get("kind")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow::anyhow!("missing kind"))?;

    let detail = value
        .get("detail")
        .cloned()
        .unwrap_or(serde_json::Value::Null);

    let kind = match kind {
        "scheduled" => EventKind::Scheduled,
        "started" => EventKind::Started,
        "completed" => EventKind::Completed,
        "yielded" => EventKind::Yielded,
        "capability_violation" => EventKind::CapabilityViolation,
        "budget_violation" => EventKind::BudgetViolation,
        "resource_violation" => EventKind::ResourceViolation,
        "snapshot_created" => EventKind::SnapshotCreated,
        "snapshot_rollback_started" => EventKind::SnapshotRollbackStarted,
        "snapshot_rollback_committed" => EventKind::SnapshotRollbackCommitted,
        "snapshot_rollback_failed" => EventKind::SnapshotRollbackFailed,
        "snapshot_integrity_verified" => EventKind::SnapshotIntegrityVerified,
        "snapshot_integrity_violation" => EventKind::SnapshotIntegrityViolation,
        "memory_read" => EventKind::MemoryRead,
        "memory_write" => EventKind::MemoryWrite,
        "checkpoint" => EventKind::Checkpoint,
        "policy_alert" => EventKind::PolicyAlert,
        "io_queued" => EventKind::IoQueued,
        "io_completed" => EventKind::IoCompleted,
        "security_violation" => EventKind::SecurityViolation,
        "custom" => {
            let label = value
                .get("label")
                .and_then(|v| v.as_str())
                .unwrap_or("custom");
            EventKind::Custom(label.to_owned())
        }
        other => anyhow::bail!("unknown event kind: {other}"),
    };

    Ok(KernelEvent {
        token_id: TokenId::new(token_id),
        kind,
        detail,
        timestamp,
    })
}

fn format_event(event: &KernelEvent) -> String {
    let token = event.token_id.raw();
    match &event.kind {
        EventKind::Scheduled => format!(
            "[{}] token {token} scheduled: {}",
            event.timestamp, event.detail
        ),
        EventKind::Started => format!("[{}] token {token} started", event.timestamp),
        EventKind::Completed => format!("[{}] token {token} completed", event.timestamp),
        EventKind::Yielded => format!("[{}] token {token} yielded", event.timestamp),
        EventKind::CapabilityViolation => format!(
            "[{}] token {token} capability violation: {}",
            event.timestamp, event.detail
        ),
        EventKind::BudgetViolation => format!(
            "[{}] token {token} budget violation: {}",
            event.timestamp, event.detail
        ),
        EventKind::ResourceViolation => format!(
            "[{}] token {token} resource violation: {}",
            event.timestamp, event.detail
        ),
        EventKind::SnapshotCreated => format!(
            "[{}] token {token} created snapshot: {}",
            event.timestamp, event.detail
        ),
        EventKind::SnapshotRollbackStarted => format!(
            "[{}] token {token} started snapshot rollback: {}",
            event.timestamp, event.detail
        ),
        EventKind::SnapshotRollbackCommitted => format!(
            "[{}] token {token} committed snapshot rollback: {}",
            event.timestamp, event.detail
        ),
        EventKind::SnapshotRollbackFailed => format!(
            "[{}] token {token} failed snapshot rollback: {}",
            event.timestamp, event.detail
        ),
        EventKind::SnapshotIntegrityVerified => format!(
            "[{}] token {token} verified snapshot ledger: {}",
            event.timestamp, event.detail
        ),
        EventKind::SnapshotIntegrityViolation => format!(
            "[{}] token {token} detected snapshot ledger violation: {}",
            event.timestamp, event.detail
        ),
        EventKind::MemoryRead => format!(
            "[{}] token {token} memory read: {}",
            event.timestamp, event.detail
        ),
        EventKind::MemoryWrite => format!(
            "[{}] token {token} memory write: {}",
            event.timestamp, event.detail
        ),
        EventKind::Checkpoint => format!(
            "[{}] token {token} checkpoint: {}",
            event.timestamp, event.detail
        ),
        EventKind::PolicyAlert => format!(
            "[{}] token {token} policy alert: {}",
            event.timestamp, event.detail
        ),
        EventKind::IoQueued => format!(
            "[{}] token {token} queued I/O: {}",
            event.timestamp, event.detail
        ),
        EventKind::IoCompleted => format!(
            "[{}] token {token} completed I/O: {}",
            event.timestamp, event.detail
        ),
        EventKind::SecurityViolation => format!(
            "[{}] token {token} security violation: {}",
            event.timestamp, event.detail
        ),
        EventKind::Custom(label) => format!(
            "[{}] token {token} {label} event: {}",
            event.timestamp, event.detail
        ),
    }
}

#[derive(Default)]
struct ReplaySummary {
    scheduled: usize,
    started: usize,
    completed: usize,
    yielded: usize,
    capability_violation: usize,
    budget_violation: usize,
    resource_violation: usize,
    security_violation: usize,
    snapshot_created: usize,
    snapshot_rollback_started: usize,
    snapshot_rollback_committed: usize,
    snapshot_rollback_failed: usize,
    snapshot_integrity_verified: usize,
    snapshot_integrity_violation: usize,
    memory_read: usize,
    memory_write: usize,
    checkpoint: usize,
    policy_alert: usize,
    io_queued: usize,
    io_completed: usize,
    custom: usize,
}

impl ReplaySummary {
    fn observe(&mut self, event: &KernelEvent) {
        match event.kind {
            EventKind::Scheduled => self.scheduled += 1,
            EventKind::Started => self.started += 1,
            EventKind::Completed => self.completed += 1,
            EventKind::Yielded => self.yielded += 1,
            EventKind::CapabilityViolation => self.capability_violation += 1,
            EventKind::BudgetViolation => self.budget_violation += 1,
            EventKind::ResourceViolation => self.resource_violation += 1,
            EventKind::SecurityViolation => self.security_violation += 1,
            EventKind::SnapshotCreated => self.snapshot_created += 1,
            EventKind::SnapshotRollbackStarted => self.snapshot_rollback_started += 1,
            EventKind::SnapshotRollbackCommitted => self.snapshot_rollback_committed += 1,
            EventKind::SnapshotRollbackFailed => self.snapshot_rollback_failed += 1,
            EventKind::SnapshotIntegrityVerified => self.snapshot_integrity_verified += 1,
            EventKind::SnapshotIntegrityViolation => self.snapshot_integrity_violation += 1,
            EventKind::MemoryRead => self.memory_read += 1,
            EventKind::MemoryWrite => self.memory_write += 1,
            EventKind::Checkpoint => self.checkpoint += 1,
            EventKind::PolicyAlert => self.policy_alert += 1,
            EventKind::IoQueued => self.io_queued += 1,
            EventKind::IoCompleted => self.io_completed += 1,
            EventKind::Custom(_) => self.custom += 1,
        }
    }

    fn render(&self, total: usize) -> String {
        format!(
            "Summary: {total} events (scheduled: {}, started: {}, completed: {}, yielded: {}, capability_violations: {}, budget_violations: {}, resource_violations: {}, security_violations: {}, snapshots: {}, rollback_started: {}, rollback_committed: {}, rollback_failed: {}, integrity_verified: {}, integrity_violations: {}, memory_reads: {}, memory_writes: {}, checkpoints: {}, policy_alerts: {}, io_queued: {}, io_completed: {}, custom: {})",
            self.scheduled,
            self.started,
            self.completed,
            self.yielded,
            self.capability_violation,
            self.budget_violation,
            self.resource_violation,
            self.security_violation,
            self.snapshot_created,
            self.snapshot_rollback_started,
            self.snapshot_rollback_committed,
            self.snapshot_rollback_failed,
            self.snapshot_integrity_verified,
            self.snapshot_integrity_violation,
            self.memory_read,
            self.memory_write,
            self.checkpoint,
            self.policy_alert,
            self.io_queued,
            self.io_completed,
            self.custom
        )
    }
}
