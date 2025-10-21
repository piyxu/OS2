use std::collections::{HashMap, HashSet};
use std::convert::TryInto;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

use ed25519_dalek::{Signature, Verifier, VerifyingKey};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModuleMetadata {
    pub name: String,
    pub hash: String,
    pub signing_key: String,
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum ModuleSecurityError {
    #[error("signing key `{0}` is not registered")]
    UnknownKey(String),
    #[error("signing key `{0}` bytes are invalid")]
    InvalidKey(String),
    #[error("module `{0}` signature is invalid")]
    InvalidSignature(String),
    #[error("module `{0}` has been revoked")]
    Revoked(String),
    #[error("module `{module}` violates signing policy: {reason}")]
    PolicyViolation { module: String, reason: String },
}

#[derive(Debug, Default)]
pub struct ModuleSecurityManager {
    trusted_keys: HashMap<String, VerifyingKey>,
    modules: HashMap<String, ModuleMetadata>,
    revoked: HashSet<String>,
    policy: ModuleSigningPolicy,
}

impl ModuleSecurityManager {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register_trusted_key(
        &mut self,
        key_id: impl Into<String>,
        key_bytes: &[u8],
    ) -> Result<(), ModuleSecurityError> {
        let key_id = key_id.into();
        let verifying = VerifyingKey::from_bytes(
            key_bytes
                .try_into()
                .map_err(|_| ModuleSecurityError::InvalidKey(key_id.clone()))?,
        )
        .map_err(|_| ModuleSecurityError::InvalidKey(key_id.clone()))?;
        self.trusted_keys.insert(key_id, verifying);
        Ok(())
    }

    pub fn register_module(
        &mut self,
        name: impl Into<String>,
        bytes: &[u8],
        signature: &[u8],
        key_id: &str,
    ) -> Result<ModuleMetadata, ModuleSecurityError> {
        let name = name.into();
        let key = self
            .trusted_keys
            .get(key_id)
            .ok_or_else(|| ModuleSecurityError::UnknownKey(key_id.to_string()))?;

        let signature = Signature::from_slice(signature)
            .map_err(|_| ModuleSecurityError::InvalidSignature(name.clone()))?;

        key.verify(bytes, &signature)
            .map_err(|_| ModuleSecurityError::InvalidSignature(name.clone()))?;

        let mut hasher = Sha256::new();
        hasher.update(bytes);
        let hash = format!("{:x}", hasher.finalize());

        let metadata = ModuleMetadata {
            name: name.clone(),
            hash: hash.clone(),
            signing_key: key_id.to_string(),
        };

        self.policy.validate(&metadata)?;
        self.modules.insert(hash.clone(), metadata.clone());
        Ok(metadata)
    }

    pub fn revoke_module(&mut self, hash: &str) {
        self.revoked.insert(hash.to_string());
    }

    pub fn ensure_allowed(&self, hash: &str) -> Result<(), ModuleSecurityError> {
        if self.revoked.contains(hash) {
            let name = self
                .modules
                .get(hash)
                .map(|m| m.name.clone())
                .unwrap_or_else(|| hash.to_string());
            return Err(ModuleSecurityError::Revoked(name));
        }
        Ok(())
    }

    pub fn metadata(&self, hash: &str) -> Option<&ModuleMetadata> {
        self.modules.get(hash)
    }

    pub fn verifying_key(&self, key_id: &str) -> Option<&VerifyingKey> {
        self.trusted_keys.get(key_id)
    }

    pub fn policy(&self) -> &ModuleSigningPolicy {
        &self.policy
    }

    pub fn policy_mut(&mut self) -> &mut ModuleSigningPolicy {
        &mut self.policy
    }
}

#[derive(Debug, Clone)]
pub struct ModuleSigningPolicy {
    enforce: bool,
    allowed_signers: HashMap<String, HashSet<String>>,
}

impl Default for ModuleSigningPolicy {
    fn default() -> Self {
        Self {
            enforce: false,
            allowed_signers: HashMap::new(),
        }
    }
}

impl ModuleSigningPolicy {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn require_signatures(&mut self, enforce: bool) {
        self.enforce = enforce;
    }

    pub fn allow_signer_for_prefix(
        &mut self,
        prefix: impl Into<String>,
        signer: impl Into<String>,
    ) {
        self.allowed_signers
            .entry(prefix.into())
            .or_default()
            .insert(signer.into());
    }

    fn validate(&self, metadata: &ModuleMetadata) -> Result<(), ModuleSecurityError> {
        if !self.enforce {
            return Ok(());
        }

        if self.allowed_signers.is_empty() {
            return Ok(());
        }

        for (prefix, signers) in &self.allowed_signers {
            if metadata.name.starts_with(prefix) {
                if signers.contains(&metadata.signing_key) {
                    return Ok(());
                }
                return Err(ModuleSecurityError::PolicyViolation {
                    module: metadata.name.clone(),
                    reason: format!(
                        "signer `{}` not allowed for prefix `{}`",
                        metadata.signing_key, prefix
                    ),
                });
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::{Signer, SigningKey};

    fn signing_key() -> SigningKey {
        SigningKey::from_bytes(&[1u8; 32])
    }

    #[test]
    fn registering_and_validating_module() {
        let mut manager = ModuleSecurityManager::new();
        let signing = signing_key();
        manager
            .register_trusted_key("primary", signing.verifying_key().as_bytes())
            .expect("registered key");

        let bytes = b"module";
        let signature = signing.sign(bytes);
        let signature_bytes = signature.to_bytes();
        let metadata = manager
            .register_module("demo", bytes, &signature_bytes, "primary")
            .expect("registered module");

        assert_eq!(metadata.name, "demo");
        assert_eq!(metadata.signing_key, "primary");
        assert!(manager.ensure_allowed(&metadata.hash).is_ok());
    }

    #[test]
    fn revocation_blocks_module() {
        let mut manager = ModuleSecurityManager::new();
        let signing = signing_key();
        manager
            .register_trusted_key("primary", signing.verifying_key().as_bytes())
            .expect("registered key");
        let bytes = b"module";
        let signature = signing.sign(bytes);
        let signature_bytes = signature.to_bytes();
        let metadata = manager
            .register_module("demo", bytes, &signature_bytes, "primary")
            .expect("registered module");

        manager.revoke_module(&metadata.hash);
        let err = manager.ensure_allowed(&metadata.hash).unwrap_err();
        assert_eq!(err, ModuleSecurityError::Revoked("demo".into()));
    }

    #[test]
    fn unknown_key_is_rejected() {
        let mut manager = ModuleSecurityManager::new();
        let err = manager
            .register_module("demo", b"data", &[0; 64], "missing")
            .unwrap_err();
        assert_eq!(err, ModuleSecurityError::UnknownKey("missing".into()));
    }

    #[test]
    fn policy_blocks_unapproved_signer() {
        let mut manager = ModuleSecurityManager::new();
        let signing = signing_key();
        manager
            .register_trusted_key("primary", signing.verifying_key().as_bytes())
            .expect("registered key");
        manager.policy_mut().require_signatures(true);
        manager
            .policy_mut()
            .allow_signer_for_prefix("core", "secondary");

        let bytes = b"module";
        let signature = signing.sign(bytes);
        let err = manager
            .register_module("core.analytics", bytes, &signature.to_bytes(), "primary")
            .unwrap_err();
        match err {
            ModuleSecurityError::PolicyViolation { module, .. } => {
                assert_eq!(module, "core.analytics");
            }
            other => panic!("unexpected error: {:?}", other),
        }
    }
}
