use ed25519_dalek::{Signature, Signer, SigningKey, Verifier};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::scheduler::{KernelToken, TokenId};
use crate::security::ModuleSecurityManager;

/// Hash of a scheduled token payload captured for auditing purposes.
fn token_payload_hash(token: &KernelToken) -> String {
    let payload = serde_json::to_vec(token).expect("token serialization");
    let mut hasher = Sha256::new();
    hasher.update(payload);
    format!("{:x}", hasher.finalize())
}

fn signature_message(payload_hash: &str, prev_hash: Option<&str>) -> Vec<u8> {
    let mut message = Vec::with_capacity(128);
    message.extend_from_slice(payload_hash.as_bytes());
    if let Some(prev) = prev_hash {
        message.push(b'|');
        message.extend_from_slice(prev.as_bytes());
    }
    message
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TokenSignature {
    pub signer: String,
    #[serde(with = "serde_bytes")]
    pub signature: Vec<u8>,
    pub payload_hash: String,
    pub previous_hash: Option<String>,
}

impl TokenSignature {
    pub fn sign(
        token: &KernelToken,
        signer: impl Into<String>,
        signing_key: &SigningKey,
        previous_hash: Option<&str>,
    ) -> Self {
        let payload_hash = token_payload_hash(token);
        let message = signature_message(&payload_hash, previous_hash);
        let signature = signing_key.sign(&message);
        Self {
            signer: signer.into(),
            signature: signature.to_bytes().to_vec(),
            payload_hash,
            previous_hash: previous_hash.map(|value| value.to_string()),
        }
    }

    pub fn entry_hash(&self) -> String {
        let mut hasher = Sha256::new();
        hasher.update(self.payload_hash.as_bytes());
        hasher.update(&self.signature);
        if let Some(prev) = &self.previous_hash {
            hasher.update(prev.as_bytes());
        }
        format!("{:x}", hasher.finalize())
    }
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum TokenSigningError {
    #[error("token {0:?} is missing a signature")]
    MissingSignature(TokenId),
    #[error("signer `{signer}` is not registered")]
    UnknownSigner { signer: String, token_id: TokenId },
    #[error("token {token_id:?} signature is invalid")]
    InvalidSignature { token_id: TokenId },
    #[error("token {token_id:?} payload hash mismatch")]
    PayloadMismatch { token_id: TokenId },
    #[error("token {token_id:?} references incorrect previous hash")]
    ChainMismatch { token_id: TokenId },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TokenSignatureRecord {
    pub token_id: u64,
    pub signer: String,
    pub payload_hash: String,
    pub previous_hash: Option<String>,
    pub entry_hash: String,
}

#[derive(Debug, Default)]
pub struct TokenSignatureLedger {
    records: Vec<TokenSignatureRecord>,
    head_hash: Option<String>,
}

impl TokenSignatureLedger {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn head_hash(&self) -> Option<&str> {
        self.head_hash.as_deref()
    }

    pub fn records(&self) -> &[TokenSignatureRecord] {
        &self.records
    }

    pub fn verify_and_append(
        &mut self,
        token: &KernelToken,
        module_security: &ModuleSecurityManager,
    ) -> Result<TokenSignatureRecord, TokenSigningError> {
        let signature = token
            .signature
            .as_ref()
            .ok_or_else(|| TokenSigningError::MissingSignature(token.id))?;

        let verifying_key = module_security
            .verifying_key(&signature.signer)
            .ok_or_else(|| TokenSigningError::UnknownSigner {
                signer: signature.signer.clone(),
                token_id: token.id,
            })?;

        let expected_payload_hash = token_payload_hash(token);
        if expected_payload_hash != signature.payload_hash {
            return Err(TokenSigningError::PayloadMismatch { token_id: token.id });
        }

        let expected_prev = self.head_hash();
        if signature.previous_hash.as_deref() != expected_prev {
            return Err(TokenSigningError::ChainMismatch { token_id: token.id });
        }

        let message = signature_message(&signature.payload_hash, expected_prev);
        let ed_signature = Signature::from_slice(&signature.signature)
            .map_err(|_| TokenSigningError::InvalidSignature { token_id: token.id })?;

        verifying_key
            .verify(&message, &ed_signature)
            .map_err(|_| TokenSigningError::InvalidSignature { token_id: token.id })?;

        let record = TokenSignatureRecord {
            token_id: token.id.raw(),
            signer: signature.signer.clone(),
            payload_hash: signature.payload_hash.clone(),
            previous_hash: signature.previous_hash.clone(),
            entry_hash: signature.entry_hash(),
        };

        self.head_hash = Some(record.entry_hash.clone());
        self.records.push(record);
        Ok(self.records.last().expect("record just pushed").clone())
    }
}

pub fn derive_signing_key(seed: u64) -> SigningKey {
    let mut hasher = Sha256::new();
    hasher.update(seed.to_le_bytes());
    let digest = hasher.finalize();
    let mut bytes = [0u8; 32];
    bytes.copy_from_slice(&digest[..32]);
    SigningKey::from_bytes(&bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scheduler::{KernelToken, TokenId, TokenKind};
    use crate::security::ModuleSecurityManager;
    use serde_json::json;

    fn signed_token(
        id: u64,
        signer: &str,
        signing_key: &SigningKey,
        previous_hash: Option<&str>,
    ) -> KernelToken {
        let token = KernelToken::new(TokenId::new(id), 5, 1, TokenKind::Plan)
            .with_context_hash(format!("token::{id}"))
            .with_goal("phase1".to_string())
            .with_payload(json!({ "id": id }));
        let signature = TokenSignature::sign(&token, signer, signing_key, previous_hash);
        token.with_signature(signature)
    }

    fn setup_security(signing_key: &SigningKey, signer: &str) -> ModuleSecurityManager {
        let mut security = ModuleSecurityManager::new();
        security
            .register_trusted_key(signer, signing_key.verifying_key().as_bytes())
            .expect("trusted key registered");
        security
    }

    #[test]
    fn ledger_appends_tokens_in_chain_order() {
        let mut ledger = TokenSignatureLedger::new();
        let signing_key = derive_signing_key(42);
        let security = setup_security(&signing_key, "kernel.system");

        let token_a = signed_token(1, "kernel.system", &signing_key, ledger.head_hash());
        let record_a = ledger
            .verify_and_append(&token_a, &security)
            .expect("first token recorded");
        assert_eq!(record_a.previous_hash, None);
        assert_eq!(ledger.head_hash(), Some(record_a.entry_hash.as_str()));

        let head = ledger.head_hash().map(|hash| hash.to_string());
        let token_b = signed_token(2, "kernel.system", &signing_key, head.as_deref());
        let record_b = ledger
            .verify_and_append(&token_b, &security)
            .expect("second token recorded");

        assert_eq!(ledger.records().len(), 2);
        assert_eq!(
            record_b.previous_hash.as_deref(),
            Some(record_a.entry_hash.as_str())
        );
        assert_eq!(ledger.head_hash(), Some(record_b.entry_hash.as_str()));
    }

    #[test]
    fn ledger_rejects_broken_hash_chain() {
        let mut ledger = TokenSignatureLedger::new();
        let signing_key = derive_signing_key(7);
        let security = setup_security(&signing_key, "kernel.system");

        let token_a = signed_token(10, "kernel.system", &signing_key, ledger.head_hash());
        ledger
            .verify_and_append(&token_a, &security)
            .expect("baseline token recorded");

        let bad_token = signed_token(11, "kernel.system", &signing_key, None);
        let err = ledger
            .verify_and_append(&bad_token, &security)
            .expect_err("hash mismatch rejected");

        assert_eq!(
            err,
            TokenSigningError::ChainMismatch {
                token_id: TokenId::new(11)
            }
        );
    }
}
