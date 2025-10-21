use anyhow::{Context, Result, anyhow};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule};
use serde::Deserialize;

use crate::capability::CapabilityLimits;
use crate::kernel::{ExecutionContext, ExecutionStatus, Kernel, TokenExecutor};
use crate::resource::ResourceRequest;
use crate::scheduler::{KernelToken, TokenBudget, TokenKind};
use crate::tooling::{ToolCatalog, ToolInvocationResult};

const DEFAULT_BUILTINS: &[&str] = &[
    "abs",
    "all",
    "any",
    "enumerate",
    "len",
    "max",
    "min",
    "range",
    "sorted",
    "sum",
];

#[derive(Debug, Clone)]
pub struct PipelineSpec {
    pub perception: String,
    pub reason: String,
    pub plan: String,
    pub act: String,
    pub reflect: String,
}

impl Default for PipelineSpec {
    fn default() -> Self {
        Self {
            perception: "perception".into(),
            reason: "reason".into(),
            plan: "plan".into(),
            act: "act".into(),
            reflect: "reflect".into(),
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct PipelineRun {
    pub perception: Option<serde_json::Value>,
    pub reason: Option<serde_json::Value>,
    pub plan: Option<serde_json::Value>,
    pub act: Option<serde_json::Value>,
    pub reflect: Option<serde_json::Value>,
}

pub struct PythonHost {
    tools: ToolCatalog,
    allowed_builtins: Vec<&'static str>,
}

impl PythonHost {
    pub fn new(
        kernel: &mut Kernel,
        adapters: Vec<Box<dyn crate::tooling::ToolAdapter>>,
    ) -> Result<Self> {
        Ok(Self {
            tools: ToolCatalog::from_adapters(kernel, adapters)?,
            allowed_builtins: DEFAULT_BUILTINS.to_vec(),
        })
    }

    pub fn with_builtins(
        kernel: &mut Kernel,
        adapters: Vec<Box<dyn crate::tooling::ToolAdapter>>,
        builtins: Vec<&'static str>,
    ) -> Result<Self> {
        Ok(Self {
            tools: ToolCatalog::from_adapters(kernel, adapters)?,
            allowed_builtins: builtins,
        })
    }

    pub fn run_agent(
        &mut self,
        kernel: &mut Kernel,
        source: &str,
        entrypoint: &str,
        context: serde_json::Value,
    ) -> Result<Vec<ToolInvocationResult>> {
        let plan_json = self.execute_python(source, entrypoint, context)?;
        let plan: Vec<PlannedInvocation> =
            serde_json::from_str(&plan_json).context("deserializing python plan")?;

        let mut results = Vec::with_capacity(plan.len());
        for invocation in plan {
            let output = self
                .tools
                .invoke(kernel, &invocation.tool, &invocation.payload)
                .with_context(|| format!("invoking tool `{}`", invocation.tool))?;
            results.push(ToolInvocationResult {
                tool: invocation.tool,
                output,
            });
        }

        Ok(results)
    }

    pub fn run_pipeline(
        &mut self,
        kernel: &mut Kernel,
        source: &str,
        spec: PipelineSpec,
        context: serde_json::Value,
    ) -> Result<PipelineRun> {
        let program = self.compile_pipeline(source, &spec)?;
        let capability = kernel.register_capability("memory_access", CapabilityLimits::unlimited());

        let mut perception = kernel.allocate_token(90, 2, TokenKind::Perception);
        let perception_context = format!("token-{}:perception", perception.id.raw());
        perception = perception
            .with_context_hash(perception_context.clone())
            .with_goal(spec.perception.clone())
            .with_payload(serde_json::json!({"stage": "perception"}))
            .with_granted_capabilities(vec![capability])
            .with_budget(TokenBudget::new(4))
            .with_resource_request(ResourceRequest {
                cpu_time: 2,
                memory_bytes: 0,
                network_bytes: 0,
                tokens: 1,
            });

        let mut reason = kernel.allocate_token(80, 2, TokenKind::Reason);
        let reason_context = format!("token-{}:reason", reason.id.raw());
        reason = reason
            .with_context_hash(reason_context.clone())
            .with_goal(spec.reason.clone())
            .with_payload(serde_json::json!({
                "stage": "reason",
                "perception_context": perception_context,
            }))
            .with_dependencies(vec![perception.id])
            .with_granted_capabilities(vec![capability])
            .with_budget(TokenBudget::new(4))
            .with_resource_request(ResourceRequest {
                cpu_time: 3,
                memory_bytes: 0,
                network_bytes: 0,
                tokens: 2,
            });

        let mut plan = kernel.allocate_token(70, 3, TokenKind::Plan);
        let plan_context = format!("token-{}:plan", plan.id.raw());
        let plan_key = format!("plan-{}", plan.id.raw());
        plan = plan
            .with_context_hash(plan_context.clone())
            .with_goal(spec.plan.clone())
            .with_payload(serde_json::json!({
                "stage": "plan",
                "reason_context": reason_context.clone(),
                "plan_key": plan_key,
            }))
            .with_dependencies(vec![reason.id])
            .with_granted_capabilities(vec![capability])
            .with_budget(TokenBudget::new(6))
            .with_resource_request(ResourceRequest {
                cpu_time: 4,
                memory_bytes: 0,
                network_bytes: 0,
                tokens: 3,
            });

        let mut act = kernel.allocate_token(60, 2, TokenKind::Act);
        let act_context = format!("token-{}:act", act.id.raw());
        act = act
            .with_context_hash(act_context)
            .with_goal(spec.act.clone())
            .with_payload(serde_json::json!({
                "stage": "act",
                "plan_key": format!("plan-{}", plan.id.raw()),
            }))
            .with_dependencies(vec![plan.id])
            .with_granted_capabilities(vec![capability])
            .with_capability(capability)
            .with_budget(TokenBudget::new(4))
            .with_resource_request(ResourceRequest {
                cpu_time: 2,
                memory_bytes: 0,
                network_bytes: 1,
                tokens: 2,
            });

        let mut reflect = kernel.allocate_token(50, 1, TokenKind::Reflect);
        let reflect_context = format!("token-{}:reflect", reflect.id.raw());
        reflect = reflect
            .with_context_hash(reflect_context)
            .with_goal(spec.reflect.clone())
            .with_payload(serde_json::json!({
                "stage": "reflect",
                "plan_key": format!("plan-{}", plan.id.raw()),
            }))
            .with_dependencies(vec![act.id])
            .with_granted_capabilities(vec![capability])
            .with_budget(TokenBudget::new(3))
            .with_resource_request(ResourceRequest {
                cpu_time: 1,
                memory_bytes: 0,
                network_bytes: 0,
                tokens: 1,
            });

        let mut executor = PipelineExecutor::new(program, context);

        kernel.submit_token(perception);
        kernel.submit_token(reason);
        kernel.submit_token(plan);
        kernel.submit_token(act);
        kernel.submit_token(reflect);

        kernel.process_until_idle(&mut executor);

        if let Some(err) = executor.error.take() {
            return Err(err);
        }

        Ok(executor.into_run())
    }

    fn execute_python(
        &self,
        source: &str,
        entrypoint: &str,
        context: serde_json::Value,
    ) -> Result<String> {
        Python::with_gil(|py| -> Result<String> {
            let json_module =
                PyModule::import_bound(py, "json").context("importing json module")?;
            let builtins =
                PyModule::import_bound(py, "builtins").context("importing builtins module")?;

            let globals = PyDict::new_bound(py);
            let allowed = PyDict::new_bound(py);
            for name in &self.allowed_builtins {
                let builtin = builtins
                    .getattr(*name)
                    .with_context(|| format!("accessing builtin `{name}`"))?;
                allowed.set_item(*name, builtin)?;
            }
            globals.set_item("__builtins__", &allowed)?;
            globals.set_item("json", &json_module)?;

            let locals = PyDict::new_bound(py);
            py.run_bound(source, Some(&globals), Some(&locals))
                .with_context(|| "executing agent source")?;

            let function = locals
                .get_item(entrypoint)?
                .ok_or_else(|| anyhow!("entrypoint `{entrypoint}` not found"))?;

            let context_str = serde_json::to_string(&context)?;
            let context_obj = json_module
                .call_method1("loads", (context_str,))
                .context("loading context into python")?;

            let plan_obj = function
                .call1((context_obj,))
                .with_context(|| format!("invoking entrypoint `{entrypoint}`"))?;
            let plan_json: String = json_module
                .call_method1("dumps", (plan_obj,))
                .context("serializing plan to json")?
                .extract()
                .context("extracting plan json string")?;

            Ok(plan_json)
        })
    }
    fn compile_pipeline(&self, source: &str, spec: &PipelineSpec) -> Result<PipelineProgram> {
        Python::with_gil(|py| -> Result<PipelineProgram> {
            let json_module =
                PyModule::import_bound(py, "json").context("importing json module")?;
            let builtins =
                PyModule::import_bound(py, "builtins").context("importing builtins module")?;

            let globals = PyDict::new_bound(py);
            let allowed = PyDict::new_bound(py);
            for name in &self.allowed_builtins {
                let builtin = builtins
                    .getattr(*name)
                    .with_context(|| format!("accessing builtin `{name}`"))?;
                allowed.set_item(*name, builtin)?;
            }
            globals.set_item("__builtins__", &allowed)?;
            globals.set_item("json", &json_module)?;

            let locals = PyDict::new_bound(py);
            py.run_bound(source, Some(&globals), Some(&locals))
                .with_context(|| "executing pipeline source")?;

            let load = |name: &str| -> Result<Py<PyAny>> {
                locals
                    .get_item(name)?
                    .ok_or_else(|| anyhow!("pipeline function `{name}` not found"))
                    .map(|value| value.into_py(py))
            };

            Ok(PipelineProgram {
                json: json_module.into_py(py),
                perception: load(&spec.perception)?,
                reason: load(&spec.reason)?,
                plan: load(&spec.plan)?,
                act: load(&spec.act)?,
                reflect: load(&spec.reflect)?,
            })
        })
    }
}

struct PipelineProgram {
    json: Py<PyAny>,
    perception: Py<PyAny>,
    reason: Py<PyAny>,
    plan: Py<PyAny>,
    act: Py<PyAny>,
    reflect: Py<PyAny>,
}

impl PipelineProgram {
    fn call_stage(
        &self,
        function: &Py<PyAny>,
        payload: serde_json::Value,
    ) -> Result<serde_json::Value> {
        Python::with_gil(|py| -> Result<serde_json::Value> {
            let json = self.json.bind(py);
            let function = function.bind(py);
            let payload_str = serde_json::to_string(&payload)?;
            let payload_obj = json
                .call_method1("loads", (payload_str,))
                .context("loading pipeline payload")?;
            let result_obj = function
                .call1((payload_obj,))
                .context("executing pipeline stage")?;
            let result_json: String = json
                .call_method1("dumps", (result_obj,))
                .context("serializing pipeline output")?
                .extract()
                .context("extracting pipeline output")?;
            let value = serde_json::from_str(&result_json).context("parsing pipeline output")?;
            Ok(value)
        })
    }
}

#[derive(Debug, Default)]
struct PipelineState {
    context: serde_json::Value,
    perception: Option<serde_json::Value>,
    reason: Option<serde_json::Value>,
    plan: Option<serde_json::Value>,
    act: Option<serde_json::Value>,
    reflect: Option<serde_json::Value>,
}

struct PipelineExecutor {
    program: PipelineProgram,
    state: PipelineState,
    error: Option<anyhow::Error>,
}

impl PipelineExecutor {
    fn new(program: PipelineProgram, context: serde_json::Value) -> Self {
        Self {
            program,
            state: PipelineState {
                context,
                ..PipelineState::default()
            },
            error: None,
        }
    }

    fn stage_input(&self) -> serde_json::Value {
        serde_json::json!({
            "context": self.state.context,
            "perception": self.state.perception,
            "reason": self.state.reason,
            "plan": self.state.plan,
            "act": self.state.act,
            "reflect": self.state.reflect,
        })
    }

    fn run_stage(&mut self, token: &KernelToken, ctx: &mut ExecutionContext) -> Result<()> {
        match token.kind {
            TokenKind::Perception => {
                let output = self
                    .program
                    .call_stage(&self.program.perception, self.stage_input())?;
                ctx.write_short_term_memory(token.id, token.context_hash.clone(), output.clone());
                self.state.perception = Some(output);
            }
            TokenKind::Reason => {
                if let Some(perception_context) = token
                    .payload
                    .get("perception_context")
                    .and_then(|value| value.as_str())
                {
                    if let Some(value) = ctx.read_short_term_memory(token.id, perception_context) {
                        if self.state.perception.is_none() {
                            self.state.perception = Some(value);
                        }
                    }
                }
                let output = self
                    .program
                    .call_stage(&self.program.reason, self.stage_input())?;
                ctx.write_short_term_memory(token.id, token.context_hash.clone(), output.clone());
                self.state.reason = Some(output);
            }
            TokenKind::Plan => {
                if let Some(reason_context) = token
                    .payload
                    .get("reason_context")
                    .and_then(|value| value.as_str())
                {
                    let _ = ctx.read_short_term_memory(token.id, reason_context);
                }
                let plan_key = token
                    .payload
                    .get("plan_key")
                    .and_then(|value| value.as_str())
                    .unwrap_or("plan");
                let output = self
                    .program
                    .call_stage(&self.program.plan, self.stage_input())?;
                ctx.write_short_term_memory(token.id, token.context_hash.clone(), output.clone());
                ctx.upsert_long_term_memory(token.id, plan_key.to_string(), output.clone());
                self.state.plan = Some(output);
            }
            TokenKind::Act => {
                if let Some(plan_key) = token
                    .payload
                    .get("plan_key")
                    .and_then(|value| value.as_str())
                {
                    let _ = ctx.read_long_term_memory(token.id, plan_key);
                }
                let output = self
                    .program
                    .call_stage(&self.program.act, self.stage_input())?;
                ctx.append_episodic_memory(token.id, output.clone());
                self.state.act = Some(output);
            }
            TokenKind::Reflect => {
                if let Some(plan_key) = token
                    .payload
                    .get("plan_key")
                    .and_then(|value| value.as_str())
                {
                    let _ = ctx.read_long_term_memory(token.id, plan_key);
                }
                let output = self
                    .program
                    .call_stage(&self.program.reflect, self.stage_input())?;
                ctx.upsert_long_term_memory(
                    token.id,
                    format!("reflect-{}", token.id.raw()),
                    output.clone(),
                );
                self.state.reflect = Some(output);
            }
        }
        Ok(())
    }

    fn into_run(self) -> PipelineRun {
        PipelineRun {
            perception: self.state.perception,
            reason: self.state.reason,
            plan: self.state.plan,
            act: self.state.act,
            reflect: self.state.reflect,
        }
    }
}

impl TokenExecutor for PipelineExecutor {
    fn execute(&mut self, token: &KernelToken, ctx: &mut ExecutionContext) -> ExecutionStatus {
        if self.error.is_none() {
            if let Err(err) = self.run_stage(token, ctx) {
                self.error = Some(err);
            }
        }
        ExecutionStatus::Completed
    }
}

#[derive(Debug, Deserialize)]
struct PlannedInvocation {
    tool: String,
    payload: serde_json::Value,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capability::CapabilityLimits;
    use crate::event_bus::EventKind;
    use crate::tooling::ToolAdapter;

    struct EchoTool {
        name: String,
        max_invocations: u64,
    }

    impl EchoTool {
        fn new(name: &str, max_invocations: u64) -> Self {
            Self {
                name: name.to_string(),
                max_invocations,
            }
        }
    }

    impl ToolAdapter for EchoTool {
        fn name(&self) -> &str {
            &self.name
        }

        fn capability_limits(&self) -> CapabilityLimits {
            CapabilityLimits {
                max_invocations: Some(self.max_invocations),
                max_tokens: Some(self.max_invocations),
            }
        }

        fn cost(&self, input: &serde_json::Value) -> u64 {
            input
                .get("tokens")
                .and_then(|value| value.as_u64())
                .unwrap_or(1)
        }

        fn invoke(&self, input: &serde_json::Value) -> Result<serde_json::Value> {
            let text = input
                .get("text")
                .and_then(|value| value.as_str())
                .unwrap_or_default();
            Ok(serde_json::json!({"echoed": text}))
        }
    }

    #[test]
    fn python_host_executes_plan_and_invokes_tools() {
        let mut kernel = Kernel::new(7);
        let adapters: Vec<Box<dyn ToolAdapter>> = vec![Box::new(EchoTool::new("echo", 4))];
        let mut host = PythonHost::new(&mut kernel, adapters).expect("host");

        let source = r#"

def agent(ctx):
    message = ctx["message"]
    return [
        {"tool": "echo", "payload": {"text": message, "tokens": 1}},
    ]
"#;
        let results = host
            .run_agent(
                &mut kernel,
                source,
                "agent",
                serde_json::json!({"message": "hello"}),
            )
            .expect("results");

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].tool, "echo");
        assert_eq!(results[0].output["echoed"], "hello");
    }

    #[test]
    fn python_host_enforces_capability_budgets() {
        let mut kernel = Kernel::new(13);
        let adapters: Vec<Box<dyn ToolAdapter>> = vec![Box::new(EchoTool::new("echo", 1))];
        let mut host = PythonHost::new(&mut kernel, adapters).expect("host");

        let source = r#"

def agent(ctx):
    return [
        {"tool": "echo", "payload": {"text": "first", "tokens": 1}},
        {"tool": "echo", "payload": {"text": "second", "tokens": 1}},
    ]
"#;

        let err = host
            .run_agent(&mut kernel, source, "agent", serde_json::json!({}))
            .expect_err("should fail");

        let message = format!("{err:#}");
        assert!(
            message.contains("capability"),
            "unexpected error message: {message}"
        );
    }

    #[test]
    fn python_host_runs_pipeline() {
        let mut kernel = Kernel::new(21);
        let adapters: Vec<Box<dyn ToolAdapter>> = Vec::new();
        let mut host = PythonHost::new(&mut kernel, adapters).expect("host");

        let source = r#"

def perception(state):
    return {"observation": state["context"]["input"]}

def reason(state):
    observation = state["perception"]["observation"]
    return {"analysis": observation.upper()}

def plan(state):
    analysis = state["reason"]["analysis"]
    return {"steps": [f"echo:{analysis}"]}

def act(state):
    steps = state["plan"]["steps"]
    return {"executed": steps}

def reflect(state):
    executed = state["act"]["executed"]
    return {"summary": executed[0]}
"#;

        let run = host
            .run_pipeline(
                &mut kernel,
                source,
                PipelineSpec::default(),
                serde_json::json!({"input": "deterministic"}),
            )
            .expect("pipeline run");

        let perception = run.perception.expect("perception result");
        let reason = run.reason.expect("reason result");
        let plan = run.plan.expect("plan result");
        let act = run.act.expect("act result");
        let reflect = run.reflect.expect("reflect result");

        assert_eq!(perception["observation"], "deterministic");
        assert_eq!(reason["analysis"], "DETERMINISTIC");
        assert_eq!(plan["steps"][0], "echo:DETERMINISTIC");
        assert_eq!(act["executed"][0], "echo:DETERMINISTIC");
        assert_eq!(reflect["summary"], "echo:DETERMINISTIC");

        assert!(
            kernel
                .memory()
                .short_term_records()
                .iter()
                .any(|record| record.context_hash.contains("perception"))
        );

        let events = kernel.drain_events();
        assert!(
            events
                .iter()
                .any(|event| matches!(event.kind, EventKind::MemoryWrite))
        );
    }
}
