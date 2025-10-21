use ed25519_dalek::{Signer, SigningKey, Verifier};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::capability::{CapabilityHandle, CapabilityLimits};
use crate::security::ModuleSecurityManager;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CapabilityGrant {
    pub handle: usize,
    pub name: String,
    pub limits: CapabilityLimits,
}

fn grant_payload_hash(grant: &CapabilityGrant) -> String {
    let payload = serde_json::to_vec(grant).expect("grant serialization");
    let mut hasher = Sha256::new();
    hasher.update(payload);
    format!("{:x}", hasher.finalize())
}

fn signature_message(payload_hash: &str, previous_hash: Option<&str>) -> Vec<u8> {
    let mut message = Vec::with_capacity(128);
    message.extend_from_slice(payload_hash.as_bytes());
    if let Some(prev) = previous_hash {
        message.push(b'|');
        message.extend_from_slice(prev.as_bytes());
    }
    message
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CapabilityGrantRecord {
    pub grant: CapabilityGrant,
    pub signer: String,
    pub payload_hash: String,
    pub previous_hash: Option<String>,
    pub entry_hash: String,
    #[serde(with = "serde_bytes")]
    pub signature: Vec<u8>,
}

impl CapabilityGrantRecord {
    fn new(
        grant: CapabilityGrant,
        signer: String,
        payload_hash: String,
        previous_hash: Option<String>,
        signature: Vec<u8>,
    ) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(payload_hash.as_bytes());
        hasher.update(&signature);
        if let Some(prev) = &previous_hash {
            hasher.update(prev.as_bytes());
        }
        let entry_hash = format!("{:x}", hasher.finalize());
        Self {
            grant,
            signer,
            payload_hash,
            previous_hash,
            entry_hash,
            signature,
        }
    }
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum CapabilityGrantError {
    #[error("signing key `{0}` is not registered")]
    UnknownSigner(String),
    #[error("capability grant signature is invalid")]
    InvalidSignature,
    #[error("capability grant payload hash mismatch")]
    PayloadMismatch,
    #[error("capability grant previous hash mismatch")]
    ChainMismatch,
    #[error("no signed grant exists for capability handle {0}")]
    MissingGrant(usize),
}

#[derive(Debug, Default, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CapabilityGrantLedger {
    records: Vec<CapabilityGrantRecord>,
    head_hash: Option<String>,
}

impl CapabilityGrantLedger {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn head_hash(&self) -> Option<&str> {
        self.head_hash.as_deref()
    }

    pub fn records(&self) -> &[CapabilityGrantRecord] {
        &self.records
    }

    pub fn record_for(&self, handle: CapabilityHandle) -> Option<&CapabilityGrantRecord> {
        self.records
            .iter()
            .rev()
            .find(|record| record.grant.handle == handle.raw())
    }

    pub fn append_signed(
        &mut self,
        name: impl Into<String>,
        handle: CapabilityHandle,
        limits: CapabilityLimits,
        signer: &str,
        signing_key: &SigningKey,
        module_security: &ModuleSecurityManager,
    ) -> Result<CapabilityGrantRecord, CapabilityGrantError> {
        let signer = signer.to_string();
        let previous_hash = self.head_hash.clone();
        let grant = CapabilityGrant {
            handle: handle.raw(),
            name: name.into(),
            limits: limits.clone(),
        };
        let payload_hash = grant_payload_hash(&grant);
        let message = signature_message(&payload_hash, previous_hash.as_deref());
        let signature = signing_key.sign(&message);

        let verifying = module_security
            .verifying_key(&signer)
            .ok_or_else(|| CapabilityGrantError::UnknownSigner(signer.clone()))?;
        verifying
            .verify(&message, &signature)
            .map_err(|_| CapabilityGrantError::InvalidSignature)?;

        let record = CapabilityGrantRecord::new(
            grant,
            signer,
            payload_hash,
            previous_hash.clone(),
            signature.to_bytes().to_vec(),
        );

        self.head_hash = Some(record.entry_hash.clone());
        self.records.push(record.clone());
        Ok(record)
    }

    pub fn ensure_signed(
        &self,
        handle: CapabilityHandle,
    ) -> Result<&CapabilityGrantRecord, CapabilityGrantError> {
        self.record_for(handle)
            .ok_or_else(|| CapabilityGrantError::MissingGrant(handle.raw()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::SigningKey;

    fn signing_key() -> SigningKey {
        SigningKey::from_bytes(&[7u8; 32])
    }

    #[test]
    fn appending_signed_grant_records_entry() {
        let signing = signing_key();
        let mut security = ModuleSecurityManager::new();
        security
            .register_trusted_key("kernel", signing.verifying_key().as_bytes())
            .expect("register key");

        let mut ledger = CapabilityGrantLedger::new();
        let handle = CapabilityHandle(1);
        let limits = CapabilityLimits::default();
        let record = ledger
            .append_signed(
                "demo",
                handle,
                limits.clone(),
                "kernel",
                &signing,
                &security,
            )
            .expect("signed grant");

        assert_eq!(record.grant.name, "demo");
        assert_eq!(record.grant.handle, handle.raw());
        assert_eq!(ledger.head_hash(), Some(record.entry_hash.as_str()));
        assert!(ledger.ensure_signed(handle).is_ok());
    }

    #[test]
    fn missing_grant_is_reported() {
        let ledger = CapabilityGrantLedger::new();
        let err = ledger.ensure_signed(CapabilityHandle(5)).unwrap_err();
        assert_eq!(err, CapabilityGrantError::MissingGrant(5));
    }
}
