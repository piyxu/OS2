use anyhow::Result;
use os2_kernel::{
    CapabilityLimits, ExecutionContext, ExecutionStatus, KernelToken, Microkernel,
    MicrokernelConfig, MicrokernelService, ResourceRequest, TokenKind,
};

struct LoggerService {
    handle: Option<os2_kernel::CapabilityHandle>,
}

impl LoggerService {
    fn new() -> Self {
        Self { handle: None }
    }
}

impl MicrokernelService for LoggerService {
    fn name(&self) -> &'static str {
        "logger"
    }

    fn install(&mut self, kernel: &mut os2_kernel::Kernel) -> Result<()> {
        let handle = kernel.register_capability(
            "logger",
            CapabilityLimits {
                max_invocations: Some(64),
                max_tokens: Some(2048),
            },
        );
        self.handle = Some(handle);
        Ok(())
    }

    fn bootstrap(&mut self, kernel: &mut os2_kernel::Kernel) -> Result<()> {
        let mut token = kernel
            .allocate_token(10, 5, TokenKind::Act)
            .with_context_hash("boot:logger")
            .with_goal("emit boot log")
            .with_payload(serde_json::json!({
                "service": self.name(),
                "message": "microkernel booted",
            }))
            .with_resource_request(ResourceRequest::cpu(5).combine(ResourceRequest::tokens(5)));

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
        ctx.emit_custom(
            token.id,
            "logger_event",
            serde_json::json!({
                "message": token
                    .payload
                    .get("message")
                    .and_then(|value| value.as_str())
                    .unwrap_or(""),
            }),
        );
        ExecutionStatus::Completed
    }
}

fn main() -> Result<()> {
    let mut microkernel = Microkernel::new(MicrokernelConfig {
        name: "os2-microkernel-demo".into(),
        seed: 7,
        boot_snapshot_label: Some("boot".into()),
        resource_limits: vec![
            (os2_kernel::ResourceType::Cpu, Some(10_000)),
            (os2_kernel::ResourceType::Tokens, Some(10_000)),
        ],
    });

    microkernel.register_service(Box::new(LoggerService::new()))?;
    microkernel.boot()?;

    let events = microkernel.run_until_idle();
    for event in events {
        println!("{} -> {}", event.timestamp, event.kind.as_str());
    }

    let image = microkernel.export_image();
    println!("boot image serialized bytes: {}", image.to_bytes()?.len());

    Ok(())
}
