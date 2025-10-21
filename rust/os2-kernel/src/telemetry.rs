use serde::{Deserialize, Serialize};

use crate::resource::ResourceUsageSnapshot;
use crate::snapshot::SnapshotLedgerEntry;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TelemetryFrame {
    pub snapshot_id: u64,
    pub entry_hash: String,
    pub previous_hash: Option<String>,
    pub resource_usage: ResourceUsageSnapshot,
    pub token_chain_head: Option<String>,
}

#[derive(Debug, Default, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TelemetrySynchronizer {
    frames: Vec<TelemetryFrame>,
}

impl TelemetrySynchronizer {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record(
        &mut self,
        entry: &SnapshotLedgerEntry,
        resource_usage: ResourceUsageSnapshot,
        token_chain_head: Option<String>,
    ) {
        let frame = TelemetryFrame {
            snapshot_id: entry.snapshot_id,
            entry_hash: entry.entry_hash.clone(),
            previous_hash: entry.previous_hash.clone(),
            resource_usage,
            token_chain_head,
        };
        self.frames.push(frame);
    }

    pub fn frames(&self) -> &[TelemetryFrame] {
        &self.frames
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn telemetry_frames_are_recorded() {
        let entry = SnapshotLedgerEntry {
            snapshot_id: 7,
            label: "test".into(),
            timestamp: 10,
            state_hash: "abc".into(),
            previous_hash: None,
            entry_hash: "def".into(),
        };
        let usage = ResourceUsageSnapshot::default();
        let mut sync = TelemetrySynchronizer::new();
        sync.record(&entry, usage, Some("token".into()));

        let frames = sync.frames();
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].snapshot_id, 7);
        assert_eq!(frames[0].entry_hash, "def");
        assert_eq!(frames[0].token_chain_head.as_deref(), Some("token"));
    }
}
