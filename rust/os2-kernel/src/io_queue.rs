use std::collections::VecDeque;

use serde::Serialize;

use crate::scheduler::TokenId;

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub enum IoOperationKind {
    Read,
    Write,
    Flush,
    Open,
    Close,
    Custom(String),
}

impl IoOperationKind {
    pub fn as_str(&self) -> &str {
        match self {
            IoOperationKind::Read => "read",
            IoOperationKind::Write => "write",
            IoOperationKind::Flush => "flush",
            IoOperationKind::Open => "open",
            IoOperationKind::Close => "close",
            IoOperationKind::Custom(_) => "custom",
        }
    }
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct IoOperationRecord {
    pub sequence: u64,
    pub token_id: TokenId,
    pub kind: IoOperationKind,
    pub detail: serde_json::Value,
    pub queued_at: u64,
}

impl IoOperationRecord {
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "sequence": self.sequence,
            "token_id": self.token_id.raw(),
            "kind": self.kind.as_str(),
            "queued_at": self.queued_at,
            "detail": self.detail.clone(),
        })
    }
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct CompletedIoOperation {
    pub sequence: u64,
    pub token_id: TokenId,
    pub kind: IoOperationKind,
    pub detail: serde_json::Value,
    pub queued_at: u64,
    pub completed_at: u64,
}

impl CompletedIoOperation {
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "sequence": self.sequence,
            "token_id": self.token_id.raw(),
            "kind": self.kind.as_str(),
            "queued_at": self.queued_at,
            "completed_at": self.completed_at,
            "detail": self.detail.clone(),
        })
    }
}

#[derive(Debug, Default)]
pub struct IoQueue {
    next_sequence: u64,
    queue: VecDeque<IoOperationRecord>,
}

impl IoQueue {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn enqueue(
        &mut self,
        token_id: TokenId,
        kind: IoOperationKind,
        detail: impl Serialize,
        queued_at: u64,
    ) -> IoOperationRecord {
        let detail = serde_json::to_value(detail).unwrap_or(serde_json::Value::Null);
        let record = IoOperationRecord {
            sequence: self.next_sequence,
            token_id,
            kind,
            detail,
            queued_at,
        };
        self.next_sequence += 1;
        self.queue.push_back(record.clone());
        record
    }

    pub fn complete_next(&mut self, completed_at: u64) -> Option<CompletedIoOperation> {
        self.queue.pop_front().map(|record| CompletedIoOperation {
            sequence: record.sequence,
            token_id: record.token_id,
            kind: record.kind,
            detail: record.detail,
            queued_at: record.queued_at,
            completed_at,
        })
    }

    pub fn iter(&self) -> impl Iterator<Item = &IoOperationRecord> {
        self.queue.iter()
    }

    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }

    pub fn len(&self) -> usize {
        self.queue.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn enqueue_assigns_monotonic_sequences() {
        let mut queue = IoQueue::new();
        let first = queue.enqueue(
            TokenId::new(1),
            IoOperationKind::Read,
            serde_json::json!({"path": "/tmp/a"}),
            10,
        );
        let second = queue.enqueue(
            TokenId::new(2),
            IoOperationKind::Write,
            serde_json::json!({"path": "/tmp/b"}),
            11,
        );

        assert_eq!(first.sequence, 0);
        assert_eq!(second.sequence, 1);
        assert_eq!(queue.len(), 2);
    }

    #[test]
    fn completion_preserves_fifo_order() {
        let mut queue = IoQueue::new();
        queue.enqueue(
            TokenId::new(1),
            IoOperationKind::Open,
            serde_json::json!({}),
            5,
        );
        queue.enqueue(
            TokenId::new(1),
            IoOperationKind::Write,
            serde_json::json!({"bytes": 3}),
            6,
        );

        let first = queue.complete_next(7).expect("first completion");
        let second = queue.complete_next(8).expect("second completion");

        assert!(queue.is_empty());
        assert_eq!(first.sequence, 0);
        assert_eq!(second.sequence, 1);
        assert!(queue.complete_next(9).is_none());
    }
}
