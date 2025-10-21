use std::collections::HashSet;

use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};

use crate::event_bus::KernelEvent;
use crate::kernel::{ExecutionContext, ExecutionStatus, Kernel, TokenExecutor};
use crate::resource::{ResourceType, ResourceUsageSnapshot};
use crate::scheduler::KernelToken;

/// Configuration for booting the PIYXU OS2 0.1.0 version microkernel image.
///
/// The configuration is serializable so future bootloaders or offline tooling
/// can embed deterministic kernel images that faithfully recreate the same
/// runtime topology on every boot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MicrokernelConfig {
    /// Human-readable identifier for the kernel build.
    pub name: String,
    /// Seed forwarded to the deterministic kernel RNG.
    pub seed: u64,
    /// Optional snapshot label emitted during boot once services are staged.
    pub boot_snapshot_label: Option<String>,
    /// Resource limits applied before services are bootstrapped.
    pub resource_limits: Vec<(ResourceType, Option<u64>)>,
}

impl Default for MicrokernelConfig {
    fn default() -> Self {
        Self {
            name: "piyxu-os2-microkernel".into(),
            seed: 0,
            boot_snapshot_label: Some("microkernel_boot".into()),
            resource_limits: Vec::new(),
        }
    }
}

/// Portable description of a bootable microkernel image.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MicrokernelImage {
    pub config: MicrokernelConfig,
    pub services: Vec<String>,
}

impl MicrokernelImage {
    /// Serialize the image to bytes suitable for embedding in a bootloader ROM
    /// or for shipping alongside offline tooling that expects a portable
    /// microkernel description.
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        Ok(serde_json::to_vec(self)?)
    }
}

/// Trait implemented by microkernel services that expose deterministic runtime
/// functionality (event bus, capability registry, snapshot store, ...).
pub trait MicrokernelService: Send {
    /// Stable service identifier used in payload routing and boot images.
    fn name(&self) -> &'static str;

    /// Install hooks into the kernel (e.g. register capabilities) prior to boot.
    fn install(&mut self, kernel: &mut Kernel) -> Result<()>;

    /// Issue bootstrap work such as scheduling init tokens after installation.
    fn bootstrap(&mut self, kernel: &mut Kernel) -> Result<()>;

    /// Whether the service is responsible for handling a token.
    fn accepts(&self, token: &KernelToken) -> bool;

    /// Execute the token using the provided kernel execution context.
    fn execute(&mut self, token: &KernelToken, ctx: &mut ExecutionContext) -> ExecutionStatus;
}

/// Deterministic PIYXU OS2 0.1.0 version microkernel orchestrating registered services.
pub struct Microkernel {
    kernel: Kernel,
    services: Vec<Box<dyn MicrokernelService>>,
    service_names: HashSet<String>,
    config: MicrokernelConfig,
    booted: bool,
}

impl Microkernel {
    /// Create a new microkernel instance with the supplied configuration.
    pub fn new(config: MicrokernelConfig) -> Self {
        Self {
            kernel: Kernel::new(config.seed),
            services: Vec::new(),
            service_names: HashSet::new(),
            config,
            booted: false,
        }
    }

    /// Register a service so it can participate in kernel bootstrapping.
    pub fn register_service(&mut self, mut service: Box<dyn MicrokernelService>) -> Result<()> {
        if !self.service_names.insert(service.name().to_string()) {
            return Err(anyhow!(
                "duplicate microkernel service registration: {}",
                service.name()
            ));
        }

        service.install(&mut self.kernel)?;
        self.services.push(service);
        Ok(())
    }

    /// Apply resource limits, emit a boot snapshot (if configured), and execute
    /// bootstrap hooks for all registered services.
    pub fn boot(&mut self) -> Result<()> {
        if self.booted {
            return Ok(());
        }

        for (resource, limit) in &self.config.resource_limits {
            self.kernel.configure_resource_limit(*resource, *limit);
        }

        if let Some(label) = &self.config.boot_snapshot_label {
            let state = serde_json::json!({
                "microkernel": self.config.name,
                "services": self
                    .services
                    .iter()
                    .map(|svc| svc.name())
                    .collect::<Vec<_>>(),
            });
            self.kernel.checkpoint(label.clone(), state);
        }

        for service in &mut self.services {
            service.bootstrap(&mut self.kernel)?;
        }

        self.kernel.run_boot_orchestrator();

        self.booted = true;
        Ok(())
    }

    /// Run the kernel until no scheduled tokens remain and return the emitted
    /// events, enabling headless integrations to capture logs.
    pub fn run_until_idle(&mut self) -> Vec<KernelEvent> {
        let mut executor = ServiceExecutor {
            services: self.services.as_mut_slice(),
        };
        self.kernel.process_until_idle(&mut executor);
        self.kernel.drain_events()
    }

    /// Snapshot the current configuration to a portable microkernel image.
    pub fn export_image(&self) -> MicrokernelImage {
        MicrokernelImage {
            config: self.config.clone(),
            services: self
                .services
                .iter()
                .map(|svc| svc.name().to_string())
                .collect(),
        }
    }

    /// Inspect cumulative resource usage since boot.
    pub fn resource_usage(&self) -> ResourceUsageSnapshot {
        self.kernel.resource_usage()
    }

    /// Access the deterministic kernel for advanced integrations (tests, host
    /// bridges, etc.).
    pub fn kernel(&self) -> &Kernel {
        &self.kernel
    }
}

struct ServiceExecutor<'a> {
    services: &'a mut [Box<dyn MicrokernelService>],
}

impl TokenExecutor for ServiceExecutor<'_> {
    fn execute(&mut self, token: &KernelToken, ctx: &mut ExecutionContext) -> ExecutionStatus {
        for service in self.services.iter_mut() {
            if service.accepts(token) {
                return service.execute(token, ctx);
            }
        }

        // If no service claimed the token, treat it as completed to keep the
        // scheduler progressing deterministically.
        ExecutionStatus::Completed
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capability::CapabilityLimits;
    use crate::event_bus::EventKind;
    use crate::resource::ResourceRequest;
    use crate::scheduler::TokenKind;

    struct EchoService {
        handle: Option<crate::capability::CapabilityHandle>,
    }

    impl EchoService {
        fn new() -> Self {
            Self { handle: None }
        }
    }

    impl MicrokernelService for EchoService {
        fn name(&self) -> &'static str {
            "echo"
        }

        fn install(&mut self, kernel: &mut Kernel) -> Result<()> {
            let handle = kernel.register_capability(
                "echo",
                CapabilityLimits {
                    max_invocations: Some(10),
                    max_tokens: Some(1024),
                },
            );
            self.handle = Some(handle);
            Ok(())
        }

        fn bootstrap(&mut self, kernel: &mut Kernel) -> Result<()> {
            let mut token = kernel
                .allocate_token(5, 3, TokenKind::Act)
                .with_context_hash("boot:echo")
                .with_goal("echo bootstrap")
                .with_payload(serde_json::json!({
                    "service": self.name(),
                    "message": "hello",
                }))
                .with_resource_request(ResourceRequest::cpu(5).combine(ResourceRequest::tokens(3)));

            if let Some(handle) = self.handle {
                token = token.with_capability(handle);
            }

            kernel.submit_token(token);
            Ok(())
        }

        fn accepts(&self, token: &KernelToken) -> bool {
            token
                .payload
                .get("service")
                .and_then(|value| value.as_str())
                == Some(self.name())
        }

        fn execute(&mut self, token: &KernelToken, ctx: &mut ExecutionContext) -> ExecutionStatus {
            let detail = serde_json::json!({
                "service": self.name(),
                "message": token
                    .payload
                    .get("message")
                    .and_then(|value| value.as_str())
                    .unwrap_or(""),
            });

            ctx.emit_custom(token.id, "service_echo", detail);
            ExecutionStatus::Completed
        }
    }

    #[test]
    fn microkernel_boots_services_and_processes_tokens() {
        let mut microkernel = Microkernel::new(MicrokernelConfig {
            name: "test-image".into(),
            seed: 42,
            boot_snapshot_label: Some("boot".into()),
            resource_limits: vec![
                (ResourceType::Cpu, Some(100)),
                (ResourceType::Tokens, Some(50)),
            ],
        });

        microkernel
            .register_service(Box::new(EchoService::new()))
            .expect("service registration succeeds");

        microkernel.boot().expect("boot succeeds");
        let events = microkernel.run_until_idle();

        assert!(!events.is_empty(), "events were emitted");
        assert!(
            events
                .iter()
                .any(|event| matches!(event.kind, EventKind::Custom(_))),
            "custom event emitted"
        );

        let image = microkernel.export_image();
        assert_eq!(image.config.name, "test-image");
        assert_eq!(image.services, vec!["echo".to_string()]);

        let usage = microkernel.resource_usage();
        assert!(usage.cpu.consumed > 0);
        assert!(usage.tokens.consumed > 0);
    }

    #[test]
    fn duplicate_services_are_rejected() {
        let mut microkernel = Microkernel::new(MicrokernelConfig::default());

        microkernel
            .register_service(Box::new(EchoService::new()))
            .expect("first registration succeeds");

        let err = microkernel
            .register_service(Box::new(EchoService::new()))
            .expect_err("duplicate registration fails");

        assert!(err.to_string().contains("duplicate microkernel service"));
    }
}
