use std::collections::{BTreeMap, HashMap, HashSet};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::scheduler::TokenId;
use crate::snapshot::SnapshotId;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ShortTermRecord {
    pub context_hash: String,
    pub value: serde_json::Value,
    pub timestamp: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LongTermRecord {
    pub key: String,
    pub value: serde_json::Value,
    pub version: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EpisodicRecord {
    pub token_id: TokenId,
    pub detail: serde_json::Value,
    pub timestamp: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct MemorySnapshot {
    short_term: Vec<ShortTermRecord>,
    long_term: Vec<LongTermRecord>,
    episodic: Vec<EpisodicRecord>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FederatedChange {
    pub origin: String,
    pub key: String,
    pub value: serde_json::Value,
    pub version: u64,
    pub capability: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RejectedChange {
    pub key: String,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct FederatedSyncReport {
    pub applied: usize,
    pub stale: Vec<String>,
    pub rejected: Vec<RejectedChange>,
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum FederationError {
    #[error("unknown replica {0}")]
    UnknownReplica(String),
    #[error("replica {replica} is not permitted to use capability {capability}")]
    CapabilityNotPermitted { replica: String, capability: String },
}

#[derive(Debug, Default, Clone)]
struct ReplicaState {
    allowed_capabilities: HashSet<String>,
    last_version: u64,
}

#[derive(Debug, Default)]
struct FederatedMemoryCoordinator {
    replicas: HashMap<String, ReplicaState>,
    lamport_clock: u64,
}

impl FederatedMemoryCoordinator {
    fn register_replica(&mut self, id: impl Into<String>, capabilities: Vec<String>) {
        let id = id.into();
        let state = ReplicaState {
            allowed_capabilities: capabilities.into_iter().collect(),
            last_version: 0,
        };
        self.replicas.insert(id, state);
    }

    fn stage_local_change(
        &mut self,
        replica: &str,
        capability: &str,
        key: String,
        version: u64,
        value: serde_json::Value,
    ) -> Result<FederatedChange, FederationError> {
        let state = self
            .replicas
            .get(replica)
            .ok_or_else(|| FederationError::UnknownReplica(replica.to_string()))?;
        if !state.allowed_capabilities.contains(capability) {
            return Err(FederationError::CapabilityNotPermitted {
                replica: replica.to_string(),
                capability: capability.to_string(),
            });
        }
        self.lamport_clock = self.lamport_clock.max(version);
        Ok(FederatedChange {
            origin: replica.to_string(),
            key,
            value,
            version,
            capability: capability.to_string(),
        })
    }

    fn apply_remote<F>(
        &mut self,
        replica: &str,
        changes: Vec<FederatedChange>,
        mut current_version: F,
    ) -> Result<(FederatedSyncReport, Vec<FederatedChange>), FederationError>
    where
        F: FnMut(&str) -> Option<u64>,
    {
        let state = self
            .replicas
            .get_mut(replica)
            .ok_or_else(|| FederationError::UnknownReplica(replica.to_string()))?;
        let mut report = FederatedSyncReport::default();
        let mut accepted = Vec::new();
        for change in changes {
            let key = change.key.clone();
            if change.capability.is_empty()
                || !state.allowed_capabilities.contains(&change.capability)
            {
                report.rejected.push(RejectedChange {
                    key,
                    reason: format!(
                        "capability {} is not permitted for replica {}",
                        change.capability, replica
                    ),
                });
                continue;
            }

            if change.version <= state.last_version {
                report.stale.push(key);
                continue;
            }

            if change.version <= current_version(&change.key).unwrap_or(0) {
                report.stale.push(key);
                continue;
            }

            state.last_version = state.last_version.max(change.version);
            self.lamport_clock = self.lamport_clock.max(change.version);
            accepted.push(change);
        }
        Ok((report, accepted))
    }
}

#[derive(Debug, Default)]
pub struct SemanticMemoryBus {
    short_term: Vec<ShortTermRecord>,
    long_term: Vec<LongTermRecord>,
    episodic: Vec<EpisodicRecord>,
    versions: BTreeMap<u64, MemorySnapshot>,
    version_counter: u64,
    federation: FederatedMemoryCoordinator,
}

impl SemanticMemoryBus {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record_short_term(&mut self, record: ShortTermRecord) {
        self.short_term.push(record);
    }

    pub fn upsert_long_term(&mut self, key: impl Into<String>, value: serde_json::Value) {
        let key = key.into();
        self.version_counter += 1;
        let version = self.version_counter;
        self.merge_long_term(key, value, version);
    }

    pub fn merge_long_term(
        &mut self,
        key: impl Into<String>,
        value: serde_json::Value,
        version: u64,
    ) -> bool {
        let key = key.into();
        if let Some(existing) = self.long_term.iter_mut().find(|entry| entry.key == key) {
            if existing.version >= version {
                return false;
            }
            existing.value = value;
            existing.version = version;
        } else {
            self.long_term.push(LongTermRecord {
                key,
                value,
                version,
            });
        }
        if version > self.version_counter {
            self.version_counter = version;
        }
        true
    }

    pub fn append_episodic(&mut self, record: EpisodicRecord) {
        self.episodic.push(record);
    }

    pub fn find_long_term(&self, key: &str) -> Option<&LongTermRecord> {
        self.long_term.iter().rev().find(|record| record.key == key)
    }

    pub fn find_latest_short_term(&self, context_hash: &str) -> Option<&ShortTermRecord> {
        self.short_term
            .iter()
            .rev()
            .find(|record| record.context_hash == context_hash)
    }

    pub fn snapshot(&mut self, id: SnapshotId) {
        self.versions.insert(
            id.raw(),
            MemorySnapshot {
                short_term: self.short_term.clone(),
                long_term: self.long_term.clone(),
                episodic: self.episodic.clone(),
            },
        );
    }

    pub fn restore(&mut self, id: SnapshotId) -> bool {
        if let Some(snapshot) = self.versions.get(&id.raw()) {
            self.short_term = snapshot.short_term.clone();
            self.long_term = snapshot.long_term.clone();
            self.episodic = snapshot.episodic.clone();
            true
        } else {
            false
        }
    }

    pub fn short_term_records(&self) -> &[ShortTermRecord] {
        &self.short_term
    }

    pub fn long_term_records(&self) -> &[LongTermRecord] {
        &self.long_term
    }

    pub fn episodic_records(&self) -> &[EpisodicRecord] {
        &self.episodic
    }

    pub fn register_replica(&mut self, id: impl Into<String>, capabilities: Vec<String>) {
        self.federation.register_replica(id, capabilities);
    }

    pub fn stage_federated_long_term(
        &mut self,
        local_replica: &str,
        capability: &str,
        key: impl Into<String>,
        value: serde_json::Value,
    ) -> Result<FederatedChange, FederationError> {
        let key = key.into();
        self.upsert_long_term(key.clone(), value.clone());
        let version = self
            .long_term
            .iter()
            .find(|entry| entry.key == key)
            .map(|entry| entry.version)
            .unwrap_or(self.version_counter);
        self.federation
            .stage_local_change(local_replica, capability, key, version, value)
    }

    pub fn apply_federated_changes(
        &mut self,
        replica: &str,
        changes: Vec<FederatedChange>,
    ) -> Result<FederatedSyncReport, FederationError> {
        let (mut report, accepted) = self.federation.apply_remote(replica, changes, |key| {
            self.long_term
                .iter()
                .find(|entry| entry.key == key)
                .map(|entry| entry.version)
        })?;

        let mut applied = 0;
        for change in accepted {
            if self.merge_long_term(change.key, change.value, change.version) {
                applied += 1;
            }
        }
        report.applied = applied;
        Ok(report)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snapshot_and_restore_memory() {
        let mut bus = SemanticMemoryBus::new();
        bus.record_short_term(ShortTermRecord {
            context_hash: "ctx".into(),
            value: serde_json::json!({"value": 1}),
            timestamp: 1,
        });
        bus.upsert_long_term("key", serde_json::json!({"v": 1}));
        bus.append_episodic(EpisodicRecord {
            token_id: TokenId::new(7),
            detail: serde_json::json!({"event": "start"}),
            timestamp: 1,
        });

        let snapshot = SnapshotId::from_raw(0);
        bus.snapshot(snapshot);

        bus.record_short_term(ShortTermRecord {
            context_hash: "ctx2".into(),
            value: serde_json::json!({"value": 2}),
            timestamp: 2,
        });

        assert!(bus.restore(snapshot));
        assert_eq!(bus.short_term_records().len(), 1);
        assert_eq!(bus.long_term_records().len(), 1);
        assert_eq!(bus.episodic_records().len(), 1);
    }

    #[test]
    fn federated_sync_merges_remote_updates() {
        let mut bus = SemanticMemoryBus::new();
        bus.register_replica("local", vec!["memory.write".into()]);
        bus.register_replica("remote", vec!["memory.write".into()]);

        let change = bus
            .stage_federated_long_term(
                "local",
                "memory.write",
                "policy",
                serde_json::json!({"score": 0.7}),
            )
            .expect("stage local change");

        assert_eq!(bus.long_term_records().len(), 1);

        let remote_change = FederatedChange {
            origin: "remote".into(),
            key: "policy".into(),
            value: serde_json::json!({"score": 0.9}),
            version: change.version + 1,
            capability: "memory.write".into(),
        };

        let report = bus
            .apply_federated_changes("remote", vec![remote_change.clone()])
            .expect("apply remote change");
        assert_eq!(report.applied, 1);
        assert!(report.stale.is_empty());
        assert!(report.rejected.is_empty());

        let record = bus
            .find_long_term("policy")
            .expect("merged long-term record");
        assert_eq!(record.version, remote_change.version);
        assert_eq!(record.value, remote_change.value);
    }

    #[test]
    fn federated_sync_rejects_untrusted_capability() {
        let mut bus = SemanticMemoryBus::new();
        bus.register_replica("local", vec!["memory.write".into()]);
        bus.register_replica("remote", vec!["memory.read".into()]);

        let change = FederatedChange {
            origin: "remote".into(),
            key: "policy".into(),
            value: serde_json::json!({"score": 0.4}),
            version: 10,
            capability: "memory.write".into(),
        };

        let report = bus
            .apply_federated_changes("remote", vec![change])
            .expect("apply remote change");
        assert_eq!(report.applied, 0);
        assert_eq!(report.rejected.len(), 1);
        assert!(bus.find_long_term("policy").is_none());
    }
}
