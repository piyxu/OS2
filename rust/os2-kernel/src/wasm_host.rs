use std::collections::HashSet;

use anyhow::{Context, Result, anyhow};
use serde::{Deserialize, Serialize};
use wasmi::core::ValType;
use wasmi::{Engine, Extern, Linker, Module, Store, Val};

use crate::capability::{CapabilityHandle, CapabilityLimits};
use crate::kernel::Kernel;
use crate::security::ModuleMetadata;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WasmSidecarConfig<'a> {
    pub name: &'a str,
    #[serde(with = "serde_bytes")]
    pub wasm_bytes: &'a [u8],
    #[serde(with = "serde_bytes")]
    pub signature: &'a [u8],
    pub signing_key_id: &'a str,
    #[serde(default = "default_allowed_exports")]
    pub allowed_exports: Vec<String>,
    #[serde(default = "default_cost_per_call")]
    pub cost_per_call: u64,
    #[serde(default = "CapabilityLimits::default")]
    pub capability_limits: CapabilityLimits,
}

fn default_allowed_exports() -> Vec<String> {
    Vec::new()
}

fn default_cost_per_call() -> u64 {
    1
}

pub struct WasmSidecar {
    module_name: String,
    metadata: ModuleMetadata,
    capability: CapabilityHandle,
    allowed_exports: HashSet<String>,
    cost_per_call: u64,
    module: Module,
    engine: Engine,
}

impl WasmSidecar {
    pub fn new(kernel: &mut Kernel, config: WasmSidecarConfig<'_>) -> Result<Self> {
        let engine = Engine::default();
        let module = Module::new(&engine, config.wasm_bytes)
            .with_context(|| anyhow!("loading WASM module `{}`", config.name))?;

        let metadata = kernel
            .register_signed_module(
                config.name,
                config.wasm_bytes,
                config.signature,
                config.signing_key_id,
            )
            .map_err(|err| anyhow!("{}", err))?;

        let capability = kernel.register_capability(
            format!("wasm::{}", config.name),
            config.capability_limits.clone(),
        );

        let allowed_exports: HashSet<String> = if config.allowed_exports.is_empty() {
            module
                .exports()
                .filter_map(|export| {
                    (matches!(export.ty(), wasmi::ExternType::Func(_)))
                        .then(|| export.name().to_string())
                })
                .collect()
        } else {
            config.allowed_exports.into_iter().collect()
        };

        if allowed_exports.is_empty() {
            return Err(anyhow!(
                "WASM sidecar `{}` has no callable exports",
                config.name
            ));
        }

        Ok(Self {
            module_name: config.name.to_string(),
            metadata,
            capability,
            allowed_exports,
            cost_per_call: config.cost_per_call,
            module,
            engine,
        })
    }

    pub fn module_metadata(&self) -> ModuleMetadata {
        self.metadata.clone()
    }

    pub fn capability(&self) -> CapabilityHandle {
        self.capability
    }

    pub fn invoke_i64(
        &self,
        kernel: &mut Kernel,
        function: &str,
        params: &[i64],
    ) -> Result<Vec<i64>> {
        if !self.allowed_exports.contains(function) {
            return Err(anyhow!(
                "function `{}` is not allowed for WASM sidecar `{}`",
                function,
                self.module_name
            ));
        }

        kernel
            .ensure_module_allowed(&self.metadata.hash)
            .map_err(|err| anyhow!("{}", err))?;

        let mut store = Store::new(&self.engine, ());
        let linker = Linker::new(&self.engine);
        let instance = linker
            .instantiate(&mut store, &self.module)
            .and_then(|pre| pre.start(&mut store))
            .with_context(|| anyhow!("instantiating WASM sidecar `{}`", self.module_name))?;

        let func = instance
            .get_export(&store, function)
            .and_then(Extern::into_func)
            .ok_or_else(|| {
                anyhow!(
                    "function `{function}` not found in WASM sidecar `{}`",
                    self.module_name
                )
            })?;

        let func_type = func.ty(&store);
        if func_type
            .params()
            .iter()
            .any(|ty| !matches!(ty, ValType::I64))
        {
            return Err(anyhow!(
                "WASM sidecar `{}` expects i64 parameters",
                self.module_name
            ));
        }

        if func_type
            .results()
            .iter()
            .any(|ty| !matches!(ty, ValType::I64))
        {
            return Err(anyhow!(
                "WASM sidecar `{}` expects i64 results",
                self.module_name
            ));
        }

        if params.len() != func_type.params().len() {
            return Err(anyhow!(
                "function `{}` expects {} parameters but {} were provided",
                function,
                func_type.params().len(),
                params.len()
            ));
        }

        let value_params: Vec<Val> = params.iter().copied().map(Val::from).collect();
        let mut results: Vec<Val> = func_type
            .results()
            .iter()
            .map(|ty| Val::default(*ty))
            .collect();

        let cost = self.cost_per_call + params.len() as u64;
        kernel
            .consume_capability(self.capability, cost)
            .map_err(|err| anyhow!("{}", err))?;

        func.call(&mut store, &value_params, &mut results)
            .with_context(|| {
                anyhow!(
                    "executing `{function}` in WASM sidecar `{}`",
                    self.module_name
                )
            })?;

        let mut outputs = Vec::with_capacity(results.len());
        for value in results {
            match value {
                Val::I64(v) => outputs.push(v),
                _ => {
                    return Err(anyhow!(
                        "WASM sidecar `{}` returned a non-i64 value",
                        self.module_name
                    ));
                }
            }
        }

        Ok(outputs)
    }

    pub fn revoke(&self, kernel: &mut Kernel) {
        kernel.revoke_module(&self.metadata.hash);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capability::CapabilityLimits;
    use crate::kernel::Kernel;
    use ed25519_dalek::{Signer, SigningKey};
    use wat::parse_str;

    fn signing_key() -> SigningKey {
        SigningKey::from_bytes(&[7u8; 32])
    }

    fn register_key(kernel: &mut Kernel, key_id: &str, key: &SigningKey) {
        kernel
            .module_security_mut()
            .register_trusted_key(key_id, key.verifying_key().as_bytes())
            .expect("key registered");
    }

    fn wasm_bytes() -> Vec<u8> {
        parse_str(
            r#"(module
                (func (export "double") (param i64) (result i64)
                    local.get 0
                    i64.const 2
                    i64.mul)
            )"#,
        )
        .expect("valid wasm")
    }

    #[test]
    fn invoking_wasm_sidecar_consumes_capability_and_returns_values() {
        let mut kernel = Kernel::new(0);
        let signing = signing_key();
        register_key(&mut kernel, "primary", &signing);

        let wasm = wasm_bytes();
        let signature = signing.sign(&wasm);
        let signature_bytes = signature.to_bytes();
        let config = WasmSidecarConfig {
            name: "math",
            wasm_bytes: &wasm,
            signature: &signature_bytes,
            signing_key_id: "primary",
            allowed_exports: vec!["double".to_string()],
            cost_per_call: 5,
            capability_limits: CapabilityLimits {
                max_invocations: Some(2),
                max_tokens: Some(20),
            },
        };

        let sidecar = WasmSidecar::new(&mut kernel, config).expect("created sidecar");

        let result = sidecar
            .invoke_i64(&mut kernel, "double", &[4])
            .expect("invocation succeeded");
        assert_eq!(result, vec![8]);

        let usage = kernel
            .capability_usage(sidecar.capability())
            .expect("usage available");
        assert_eq!(usage.invocations, 1);
        assert_eq!(usage.tokens, 6);
    }

    #[test]
    fn revoking_module_blocks_future_invocations() {
        let mut kernel = Kernel::new(0);
        let signing = signing_key();
        register_key(&mut kernel, "primary", &signing);

        let wasm = wasm_bytes();
        let signature = signing.sign(&wasm);
        let signature_bytes = signature.to_bytes();
        let config = WasmSidecarConfig {
            name: "math",
            wasm_bytes: &wasm,
            signature: &signature_bytes,
            signing_key_id: "primary",
            allowed_exports: vec!["double".to_string()],
            cost_per_call: 1,
            capability_limits: CapabilityLimits {
                max_invocations: Some(5),
                max_tokens: Some(10),
            },
        };

        let sidecar = WasmSidecar::new(&mut kernel, config).expect("created sidecar");
        sidecar.revoke(&mut kernel);

        let err = sidecar
            .invoke_i64(&mut kernel, "double", &[2])
            .expect_err("revocation enforced");
        assert!(err.to_string().contains("revoked"));
    }

    #[test]
    fn enforcing_capability_limits() {
        let mut kernel = Kernel::new(0);
        let signing = signing_key();
        register_key(&mut kernel, "primary", &signing);

        let wasm = wasm_bytes();
        let signature = signing.sign(&wasm);
        let signature_bytes = signature.to_bytes();
        let config = WasmSidecarConfig {
            name: "math",
            wasm_bytes: &wasm,
            signature: &signature_bytes,
            signing_key_id: "primary",
            allowed_exports: vec!["double".to_string()],
            cost_per_call: 1,
            capability_limits: CapabilityLimits {
                max_invocations: Some(1),
                max_tokens: Some(2),
            },
        };

        let sidecar = WasmSidecar::new(&mut kernel, config).expect("created sidecar");
        assert!(sidecar.invoke_i64(&mut kernel, "double", &[1]).is_ok());
        let err = sidecar
            .invoke_i64(&mut kernel, "double", &[1])
            .expect_err("invocation limited");
        assert!(
            err.to_string()
                .contains("capability invocation budget exhausted")
        );
    }
}
