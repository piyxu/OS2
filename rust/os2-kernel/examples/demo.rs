use os2_kernel::{
    CapabilityLimits, EventKind, Kernel, KernelToken, ResourceRequest, TokenExecutor, TokenKind,
};

struct DemoExecutor;

impl TokenExecutor for DemoExecutor {
    fn execute(
        &mut self,
        token: &KernelToken,
        ctx: &mut os2_kernel::ExecutionContext,
    ) -> os2_kernel::ExecutionStatus {
        ctx.emit_custom(
            token.id,
            "demo",
            serde_json::json!({
                "message": format!("token {:?} executed", token.kind),
            }),
        );
        os2_kernel::ExecutionStatus::Completed
    }
}

fn main() {
    let mut kernel = Kernel::new(1234);
    let capability = kernel.register_capability(
        "demo",
        CapabilityLimits {
            max_invocations: Some(10),
            max_tokens: Some(100),
        },
    );

    let token = kernel
        .allocate_token(5, 10, TokenKind::Reason)
        .with_capability(capability)
        .with_payload(serde_json::json!({"goal": "demo"}))
        .with_resource_request(ResourceRequest {
            cpu_time: 5,
            memory_bytes: 0,
            network_bytes: 0,
            tokens: 5,
        });

    kernel.submit_token(token);
    let mut executor = DemoExecutor;
    kernel.process_until_idle(&mut executor);

    for event in kernel.drain_events() {
        match event.kind {
            EventKind::Started => println!("token {:?} started", event.token_id.raw()),
            EventKind::Completed => println!("token {:?} completed", event.token_id.raw()),
            EventKind::Custom(label) => println!(
                "token {:?} emitted custom event {}: {}",
                event.token_id.raw(),
                label,
                event.detail
            ),
            _ => println!("{event:?}"),
        }
    }
}
