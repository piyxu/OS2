use std::collections::VecDeque;

use serde::Serialize;
use serde_json::json;

use crate::clock::DeterministicClock;
use crate::scheduler::TokenId;

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct KernelEvent {
    pub token_id: TokenId,
    pub kind: EventKind,
    pub detail: serde_json::Value,
    pub timestamp: u64,
}

impl KernelEvent {
    pub fn to_json(&self) -> serde_json::Value {
        let mut base = json!({
            "token_id": self.token_id.raw(),
            "timestamp": self.timestamp,
            "kind": self.kind.as_str(),
            "detail": self.detail,
        });

        if let EventKind::Custom(label) = &self.kind {
            if let serde_json::Value::Object(ref mut map) = base {
                map.insert("label".into(), serde_json::Value::String(label.clone()));
            }
        }

        base
    }
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub enum EventKind {
    Scheduled,
    Started,
    Completed,
    Yielded,
    CapabilityViolation,
    BudgetViolation,
    ResourceViolation,
    SnapshotCreated,
    SnapshotRollbackStarted,
    SnapshotRollbackCommitted,
    SnapshotRollbackFailed,
    SnapshotIntegrityVerified,
    SnapshotIntegrityViolation,
    MemoryRead,
    Checkpoint,
    PolicyAlert,
    MemoryWrite,
    IoQueued,
    IoCompleted,
    SecurityViolation,
    Custom(String),
}

impl EventKind {
    pub fn as_str(&self) -> &str {
        match self {
            EventKind::Scheduled => "scheduled",
            EventKind::Started => "started",
            EventKind::Completed => "completed",
            EventKind::Yielded => "yielded",
            EventKind::CapabilityViolation => "capability_violation",
            EventKind::BudgetViolation => "budget_violation",
            EventKind::ResourceViolation => "resource_violation",
            EventKind::SnapshotCreated => "snapshot_created",
            EventKind::SnapshotRollbackStarted => "snapshot_rollback_started",
            EventKind::SnapshotRollbackCommitted => "snapshot_rollback_committed",
            EventKind::SnapshotRollbackFailed => "snapshot_rollback_failed",
            EventKind::SnapshotIntegrityVerified => "snapshot_integrity_verified",
            EventKind::SnapshotIntegrityViolation => "snapshot_integrity_violation",
            EventKind::MemoryRead => "memory_read",
            EventKind::MemoryWrite => "memory_write",
            EventKind::Checkpoint => "checkpoint",
            EventKind::PolicyAlert => "policy_alert",
            EventKind::IoQueued => "io_queued",
            EventKind::IoCompleted => "io_completed",
            EventKind::SecurityViolation => "security_violation",
            EventKind::Custom(_) => "custom",
        }
    }
}

#[derive(Debug, Default)]
pub struct EventBus {
    queue: VecDeque<KernelEvent>,
}

impl EventBus {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn publish(&mut self, clock: &mut DeterministicClock, event: EventBuilder) {
        let timestamp = clock.tick();
        self.queue.push_back(event.into_event(timestamp));
    }

    pub fn drain(&mut self) -> Vec<KernelEvent> {
        self.queue.drain(..).collect()
    }
}

pub struct EventBuilder {
    token_id: TokenId,
    kind: EventKind,
    detail: serde_json::Value,
}

impl EventBuilder {
    pub fn new(token_id: TokenId, kind: EventKind) -> Self {
        Self {
            token_id,
            kind,
            detail: serde_json::Value::Null,
        }
    }

    pub fn detail(mut self, value: impl Serialize) -> Self {
        self.detail = serde_json::to_value(value).unwrap_or(serde_json::Value::Null);
        self
    }

    fn into_event(self, timestamp: u64) -> KernelEvent {
        KernelEvent {
            token_id: self.token_id,
            kind: self.kind,
            detail: self.detail,
            timestamp,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn publishing_events_assigns_monotonic_timestamps() {
        let mut bus = EventBus::new();
        let mut clock = DeterministicClock::new();

        bus.publish(
            &mut clock,
            EventBuilder::new(TokenId::new(1), EventKind::Started).detail("first"),
        );
        bus.publish(
            &mut clock,
            EventBuilder::new(TokenId::new(1), EventKind::Completed).detail("second"),
        );

        let events = bus.drain();
        assert_eq!(events.len(), 2);
        assert!(events[1].timestamp > events[0].timestamp);
    }

    #[test]
    fn kernel_events_convert_to_structured_json() {
        let event = KernelEvent {
            token_id: TokenId::new(7),
            kind: EventKind::Custom("execution".into()),
            detail: serde_json::json!({"ok": true}),
            timestamp: 33,
        };

        let json = event.to_json();
        assert_eq!(json["token_id"], 7);
        assert_eq!(json["timestamp"], 33);
        assert_eq!(json["kind"], "custom");
        assert_eq!(json["label"], "execution");
        assert_eq!(json["detail"], serde_json::json!({"ok": true}));
    }
}
