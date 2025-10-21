use anyhow::Result;
use os2_kernel::{Kernel, PipelineSpec, PythonHost, ResourceType};

fn main() -> Result<()> {
    let mut kernel = Kernel::new(2025);
    kernel.configure_resource_limit(ResourceType::Cpu, Some(12));
    kernel.configure_resource_limit(ResourceType::Tokens, Some(9));
    let adapters: Vec<Box<dyn os2_kernel::ToolAdapter>> = Vec::new();
    let mut host = PythonHost::new(&mut kernel, adapters)?;

    let source = r#"

def perception(state):
    return {"observation": state["context"].get("input", "")}

def reason(state):
    observation = state["perception"]["observation"]
    return {"analysis": observation.upper()}

def plan(state):
    analysis = state["reason"]["analysis"]
    return {"steps": [f"notify:{analysis}"]}

def act(state):
    steps = state["plan"]["steps"]
    return {"executed": steps}

def reflect(state):
    executed = state["act"]["executed"]
    return {"summary": executed[0]}
"#;

    let run = host.run_pipeline(
        &mut kernel,
        source,
        PipelineSpec::default(),
        serde_json::json!({"input": "pipeline demo"}),
    )?;

    println!("Perception -> {:?}", run.perception);
    println!("Reason -> {:?}", run.reason);
    println!("Plan -> {:?}", run.plan);
    println!("Act -> {:?}", run.act);
    println!("Reflect -> {:?}", run.reflect);

    for event in kernel.drain_events() {
        println!("event: {:?}", event);
    }

    Ok(())
}
