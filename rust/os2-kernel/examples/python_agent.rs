use anyhow::Result;
use os2_kernel::{CapabilityLimits, Kernel, PythonHost, ToolAdapter};

struct UppercaseTool;

impl ToolAdapter for UppercaseTool {
    fn name(&self) -> &str {
        "uppercase"
    }

    fn capability_limits(&self) -> CapabilityLimits {
        CapabilityLimits {
            max_invocations: Some(4),
            max_tokens: Some(4),
        }
    }

    fn invoke(&self, input: &serde_json::Value) -> Result<serde_json::Value> {
        let text = input
            .get("text")
            .and_then(|value| value.as_str())
            .unwrap_or_default();
        Ok(serde_json::json!({ "text": text.to_uppercase() }))
    }
}

fn main() -> Result<()> {
    let mut kernel = Kernel::new(1234);
    let adapters: Vec<Box<dyn ToolAdapter>> = vec![Box::new(UppercaseTool)];
    let mut host = PythonHost::new(&mut kernel, adapters)?;

    let source = r#"

def agent(ctx):
    message = ctx.get("message", "")
    return [
        {"tool": "uppercase", "payload": {"text": message, "tokens": 1}}
    ]
"#;

    let plan = host.run_agent(
        &mut kernel,
        source,
        "agent",
        serde_json::json!({"message": "os2 kernel"}),
    )?;

    for invocation in plan {
        println!("{} => {}", invocation.tool, invocation.output);
    }

    Ok(())
}
