use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::clock::DeterministicClock;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SnapshotId(u64);

impl SnapshotId {
    pub fn raw(&self) -> u64 {
        self.0
    }

    pub fn from_raw(id: u64) -> Self {
        Self(id)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SnapshotState {
    pub label: String,
    pub state: serde_json::Value,
    pub timestamp: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SnapshotRollbackRecord {
    pub from_snapshot: Option<SnapshotId>,
    pub to_snapshot: SnapshotId,
    pub committed_snapshot: SnapshotId,
    pub entry_hash: String,
    pub timestamp: u64,
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum SnapshotRollbackError {
    #[error("no active snapshot available for rollback")]
    NoActiveSnapshot,
    #[error("snapshot {0:?} not found")]
    UnknownSnapshot(SnapshotId),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SnapshotLedgerIntegrityFailure {
    pub kind: SnapshotLedgerIntegrityFailureKind,
    pub entry: SnapshotLedgerEntry,
    pub expected_hash: Option<String>,
    pub expected_previous: Option<String>,
}

impl SnapshotLedgerIntegrityFailure {
    fn new_entry_mismatch(entry: &SnapshotLedgerEntry, expected: String) -> Self {
        Self {
            kind: SnapshotLedgerIntegrityFailureKind::EntryHashMismatch,
            entry: entry.clone(),
            expected_hash: Some(expected),
            expected_previous: entry.previous_hash.clone(),
        }
    }

    fn new_previous_mismatch(entry: &SnapshotLedgerEntry, expected_previous: Option<&str>) -> Self {
        Self {
            kind: SnapshotLedgerIntegrityFailureKind::PreviousHashMismatch,
            entry: entry.clone(),
            expected_hash: None,
            expected_previous: expected_previous.map(str::to_string),
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum SnapshotLedgerIntegrityFailureKind {
    EntryHashMismatch,
    PreviousHashMismatch,
}

impl SnapshotLedgerIntegrityFailureKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            SnapshotLedgerIntegrityFailureKind::EntryHashMismatch => "entry_hash_mismatch",
            SnapshotLedgerIntegrityFailureKind::PreviousHashMismatch => "previous_hash_mismatch",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SnapshotLedgerIntegrityReport {
    pub is_valid: bool,
    pub checked_entries: usize,
    pub head_hash: Option<String>,
    pub failure: Option<SnapshotLedgerIntegrityFailure>,
}

impl SnapshotLedgerIntegrityReport {
    pub fn valid(head_hash: Option<String>, checked_entries: usize) -> Self {
        Self {
            is_valid: true,
            checked_entries,
            head_hash,
            failure: None,
        }
    }

    pub fn invalid(
        head_hash: Option<String>,
        checked_entries: usize,
        failure: SnapshotLedgerIntegrityFailure,
    ) -> Self {
        Self {
            is_valid: false,
            checked_entries,
            head_hash,
            failure: Some(failure),
        }
    }
}

#[derive(Debug, Default)]
pub struct SnapshotEngine {
    clock: DeterministicClock,
    next_id: u64,
    snapshots: HashMap<SnapshotId, SnapshotState>,
    ledger: SnapshotLedger,
    state_archive: HashMap<String, SnapshotState>,
    diff_ledger: SnapshotDiffLedger,
    active_snapshot: Option<SnapshotId>,
    rollback_log: Vec<SnapshotRollbackRecord>,
    last_integrity: Option<SnapshotLedgerIntegrityReport>,
}

impl SnapshotEngine {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn checkpoint(&mut self, label: impl Into<String>, state: serde_json::Value) -> SnapshotId {
        let id = SnapshotId(self.next_id);
        self.next_id += 1;
        let timestamp = self.clock.tick();
        let snapshot_state = SnapshotState {
            label: label.into(),
            state,
            timestamp,
        };
        let entry = self.ledger.append(id, &snapshot_state).clone();
        self.state_archive
            .insert(entry.entry_hash.clone(), snapshot_state.clone());
        let previous_state = entry
            .previous_hash
            .as_ref()
            .and_then(|hash| self.state_archive.get(hash));
        self.diff_ledger
            .append(&entry, previous_state, &snapshot_state);
        self.snapshots.insert(id, snapshot_state);
        self.active_snapshot = Some(id);
        id
    }

    pub fn restore(&self, id: SnapshotId) -> Option<&SnapshotState> {
        self.snapshots.get(&id)
    }

    pub fn latest(&self) -> Option<(&SnapshotId, &SnapshotState)> {
        self.active_snapshot
            .and_then(|id| self.snapshots.get_key_value(&id))
    }

    pub fn active_snapshot(&self) -> Option<SnapshotId> {
        self.active_snapshot
    }

    pub fn ledger_entries(&self) -> &[SnapshotLedgerEntry] {
        self.ledger.entries()
    }

    pub fn ledger_head(&self) -> Option<&SnapshotLedgerEntry> {
        self.ledger.head()
    }

    pub fn verify_ledger(&self) -> bool {
        self.ledger.integrity_report().is_valid
    }

    pub fn ledger_integrity(&self) -> SnapshotLedgerIntegrityReport {
        self.ledger.integrity_report()
    }

    pub fn check_integrity(&mut self) -> SnapshotLedgerIntegrityReport {
        let report = self.ledger_integrity();
        self.last_integrity = Some(report.clone());
        report
    }

    pub fn last_integrity(&self) -> Option<&SnapshotLedgerIntegrityReport> {
        self.last_integrity.as_ref()
    }

    pub fn load_from_ledger(&self, entry_hash: &str) -> Option<SnapshotState> {
        self.state_archive.get(entry_hash).cloned()
    }

    pub fn diff_entries(&self) -> &[SnapshotDiffEntry] {
        self.diff_ledger.entries()
    }

    pub fn rollback(
        &mut self,
        target: SnapshotId,
    ) -> Result<SnapshotRollbackRecord, SnapshotRollbackError> {
        let target_state = self
            .snapshots
            .get(&target)
            .cloned()
            .ok_or(SnapshotRollbackError::UnknownSnapshot(target))?;

        let from_snapshot = self
            .active_snapshot
            .ok_or(SnapshotRollbackError::NoActiveSnapshot)?;

        let timestamp = self.clock.tick();
        let rollback_id = SnapshotId(self.next_id);
        self.next_id += 1;

        let rollback_state = SnapshotState {
            label: format!("rollback::{}#{}", target_state.label, target.raw()),
            state: target_state.state.clone(),
            timestamp,
        };

        let entry = self.ledger.append(rollback_id, &rollback_state).clone();
        self.state_archive
            .insert(entry.entry_hash.clone(), rollback_state.clone());
        let previous_state = entry
            .previous_hash
            .as_ref()
            .and_then(|hash| self.state_archive.get(hash));
        self.diff_ledger
            .append(&entry, previous_state, &rollback_state);

        self.snapshots.insert(rollback_id, rollback_state);
        self.active_snapshot = Some(rollback_id);

        let record = SnapshotRollbackRecord {
            from_snapshot: Some(from_snapshot),
            to_snapshot: target,
            committed_snapshot: rollback_id,
            entry_hash: entry.entry_hash,
            timestamp,
        };
        self.rollback_log.push(record.clone());
        Ok(record)
    }

    pub fn rollback_records(&self) -> &[SnapshotRollbackRecord] {
        &self.rollback_log
    }

    #[cfg(test)]
    pub fn inject_ledger_corruption(&mut self, index: usize, entry_hash: impl Into<String>) {
        if let Some(entry) = self.ledger.entries.get_mut(index) {
            entry.entry_hash = entry_hash.into();
        }
    }
}

fn hash_state(state: &serde_json::Value) -> String {
    let mut hasher = Sha256::new();
    if let Ok(bytes) = serde_json::to_vec(state) {
        hasher.update(bytes);
    }
    format!("{:x}", hasher.finalize())
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SnapshotLedgerEntry {
    pub snapshot_id: u64,
    pub label: String,
    pub timestamp: u64,
    pub state_hash: String,
    pub previous_hash: Option<String>,
    pub entry_hash: String,
}

#[derive(Debug, Default, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SnapshotLedger {
    entries: Vec<SnapshotLedgerEntry>,
    head_hash: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SnapshotDiffEntry {
    pub snapshot_id: u64,
    pub entry_hash: String,
    pub previous_hash: Option<String>,
    pub added: serde_json::Value,
    pub removed_keys: Vec<String>,
    pub replaced: bool,
}

#[derive(Debug, Default, Clone, Serialize, Deserialize, PartialEq)]
pub struct SnapshotDiffLedger {
    entries: Vec<SnapshotDiffEntry>,
}

impl SnapshotLedger {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn append(&mut self, id: SnapshotId, state: &SnapshotState) -> &SnapshotLedgerEntry {
        let state_hash = hash_state(&state.state);
        let previous = self.head_hash.clone();
        let mut hasher = Sha256::new();
        hasher.update(id.raw().to_le_bytes());
        hasher.update(state_hash.as_bytes());
        if let Some(prev) = &previous {
            hasher.update(prev.as_bytes());
        }
        let entry_hash = format!("{:x}", hasher.finalize());

        let entry = SnapshotLedgerEntry {
            snapshot_id: id.raw(),
            label: state.label.clone(),
            timestamp: state.timestamp,
            state_hash,
            previous_hash: previous.clone(),
            entry_hash: entry_hash.clone(),
        };

        self.head_hash = Some(entry_hash);
        self.entries.push(entry);
        self.entries.last().expect("entry appended")
    }

    pub fn entries(&self) -> &[SnapshotLedgerEntry] {
        &self.entries
    }

    pub fn head(&self) -> Option<&SnapshotLedgerEntry> {
        self.entries.last()
    }

    pub fn head_hash(&self) -> Option<&str> {
        self.head_hash.as_deref()
    }

    pub fn integrity_report(&self) -> SnapshotLedgerIntegrityReport {
        let mut previous: Option<&str> = None;
        let mut checked = 0usize;

        for entry in &self.entries {
            checked += 1;
            let mut hasher = Sha256::new();
            hasher.update(entry.snapshot_id.to_le_bytes());
            hasher.update(entry.state_hash.as_bytes());
            if let Some(prev) = previous {
                hasher.update(prev.as_bytes());
            }
            let computed = format!("{:x}", hasher.finalize());
            if computed != entry.entry_hash {
                let failure = SnapshotLedgerIntegrityFailure::new_entry_mismatch(entry, computed);
                return SnapshotLedgerIntegrityReport::invalid(
                    self.head_hash.clone(),
                    checked,
                    failure,
                );
            }
            if entry.previous_hash.as_deref() != previous {
                let failure =
                    SnapshotLedgerIntegrityFailure::new_previous_mismatch(entry, previous);
                return SnapshotLedgerIntegrityReport::invalid(
                    self.head_hash.clone(),
                    checked,
                    failure,
                );
            }
            previous = Some(entry.entry_hash.as_str());
        }

        SnapshotLedgerIntegrityReport::valid(self.head_hash.clone(), checked)
    }
}

impl SnapshotDiffLedger {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn append(
        &mut self,
        entry: &SnapshotLedgerEntry,
        previous: Option<&SnapshotState>,
        current: &SnapshotState,
    ) {
        let (added, removed_keys, replaced) = diff_states(previous, current);
        let diff_entry = SnapshotDiffEntry {
            snapshot_id: entry.snapshot_id,
            entry_hash: entry.entry_hash.clone(),
            previous_hash: entry.previous_hash.clone(),
            added,
            removed_keys,
            replaced,
        };
        self.entries.push(diff_entry);
    }

    pub fn entries(&self) -> &[SnapshotDiffEntry] {
        &self.entries
    }
}

fn diff_states(
    previous: Option<&SnapshotState>,
    current: &SnapshotState,
) -> (serde_json::Value, Vec<String>, bool) {
    let prev_state = previous.map(|state| &state.state);
    match (prev_state, &current.state) {
        (Some(serde_json::Value::Object(prev)), serde_json::Value::Object(curr)) => {
            let mut added_map = serde_json::Map::new();
            for (key, value) in curr {
                if prev.get(key) != Some(value) {
                    added_map.insert(key.clone(), value.clone());
                }
            }
            let removed_keys = prev
                .keys()
                .filter(|key| !curr.contains_key(*key))
                .cloned()
                .collect::<Vec<_>>();
            (serde_json::Value::Object(added_map), removed_keys, false)
        }
        (Some(serde_json::Value::Array(prev)), serde_json::Value::Array(curr)) => {
            let replaced = prev != curr;
            (serde_json::Value::Array(curr.clone()), Vec::new(), replaced)
        }
        (Some(_), _) => (current.state.clone(), Vec::new(), true),
        (None, _) => (current.state.clone(), Vec::new(), true),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checkpoints_can_be_restored() {
        let mut engine = SnapshotEngine::new();
        let snapshot_id = engine.checkpoint("initial", serde_json::json!({"value": 1}));
        let restored = engine.restore(snapshot_id).unwrap();
        assert_eq!(restored.label, "initial");
        assert_eq!(restored.state["value"], 1);
    }

    #[test]
    fn ledger_records_hash_chain() {
        let mut engine = SnapshotEngine::new();
        let first = engine.checkpoint("first", serde_json::json!({"value": 1}));
        let second = engine.checkpoint("second", serde_json::json!({"value": 2}));

        let entries = engine.ledger_entries();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].snapshot_id, first.raw());
        assert_eq!(entries[1].snapshot_id, second.raw());
        assert_eq!(
            entries[1].previous_hash.as_deref(),
            Some(entries[0].entry_hash.as_str())
        );
        assert!(engine.verify_ledger());

        let head = engine.ledger_head().expect("head available");
        let loaded = engine
            .load_from_ledger(head.entry_hash.as_str())
            .expect("loaded state");
        assert_eq!(loaded.label, "second");
    }

    #[test]
    fn diff_ledger_tracks_changes() {
        let mut engine = SnapshotEngine::new();
        engine.checkpoint("base", serde_json::json!({"value": 1, "keep": true}));
        engine.checkpoint("update", serde_json::json!({"value": 2, "new": "item"}));

        let diffs = engine.diff_entries();
        assert_eq!(diffs.len(), 2);
        let latest = diffs.last().expect("latest diff");
        assert_eq!(latest.added["value"], 2);
        assert!(latest.removed_keys.contains(&"keep".to_string()));
    }

    #[test]
    fn rollback_clones_target_state_with_new_snapshot() {
        let mut engine = SnapshotEngine::new();
        let initial = engine.checkpoint("initial", serde_json::json!({"value": 1}));
        let updated = engine.checkpoint("updated", serde_json::json!({"value": 2}));

        let record = engine.rollback(initial).expect("rollback succeeded");

        assert_eq!(record.to_snapshot, initial);
        assert_eq!(record.from_snapshot, Some(updated));
        assert_ne!(record.committed_snapshot, initial);
        assert!(record.entry_hash.len() > 0);

        let (active_id, active_state) = engine.latest().expect("active snapshot");
        assert_eq!(*active_id, record.committed_snapshot);
        assert_eq!(active_state.state, serde_json::json!({"value": 1}));

        let ledger_head = engine.ledger_head().expect("ledger head");
        assert_eq!(ledger_head.snapshot_id, record.committed_snapshot.raw());
    }

    #[test]
    fn integrity_check_detects_corruption() {
        let mut engine = SnapshotEngine::new();
        engine.checkpoint("first", serde_json::json!({"value": 1}));
        let report = engine.check_integrity();
        assert!(report.is_valid);
        assert!(engine.last_integrity().is_some());

        engine.ledger.entries[0].entry_hash = "corrupt".into();
        let report = engine.check_integrity();
        assert!(!report.is_valid);
        let failure = report.failure.expect("failure present");
        assert!(matches!(
            failure.kind,
            SnapshotLedgerIntegrityFailureKind::EntryHashMismatch
        ));
        assert_eq!(failure.entry.snapshot_id, 0);
    }

    #[test]
    fn rollback_errors_when_snapshot_missing() {
        let mut engine = SnapshotEngine::new();
        engine.checkpoint("baseline", serde_json::json!({"value": 1}));
        let missing = SnapshotId::from_raw(99);

        let err = engine.rollback(missing).expect_err("missing snapshot");
        assert_eq!(err, SnapshotRollbackError::UnknownSnapshot(missing));
    }
}
