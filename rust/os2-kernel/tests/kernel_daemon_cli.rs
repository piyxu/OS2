use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Command;

use serde_json::json;

fn write_config() -> PathBuf {
    let mut path = std::env::temp_dir();
    let unique = format!("os2_kernel_daemon_config_{}.json", std::process::id());
    path.push(unique);

    let config = json!({
        "seed": 7,
        "capabilities": [
            {"name": "plan", "limits": {"max_invocations": 5, "max_tokens": 50}}
        ],
        "tokens": [
            {
                "priority": 10,
                "cost": 3,
                "kind": "Reason",
                "capability": "plan",
                "payload": {
                    "actions": [
                        {"type": "emit", "label": "plan", "detail": {"step": 1}},
                        {"type": "checkpoint", "label": "state", "state": {"value": "snapshot"}}
                    ]
                }
            }
        ]
    });

    let mut file = fs::File::create(&path).expect("create config");
    write!(file, "{}", serde_json::to_string_pretty(&config).unwrap()).expect("write config");
    path
}

#[test]
fn kernel_daemon_runs_from_config() {
    let config_path = write_config();
    let mut events_path = std::env::temp_dir();
    events_path.push(format!(
        "os2_kernel_daemon_events_{}_.jsonl",
        std::process::id()
    ));

    let exe = env!("CARGO_BIN_EXE_kernel_daemon");
    let output = Command::new(exe)
        .arg(&config_path)
        .arg(&events_path)
        .output()
        .expect("run kernel daemon");

    fs::remove_file(&config_path).ok();

    assert!(
        output.status.success(),
        "kernel daemon failed: {:?}",
        output
    );
    let stdout = String::from_utf8(output.stdout).expect("stdout utf8");
    assert!(stdout.contains("Kernel daemon summary"));
    assert!(stdout.contains("started"));
    assert!(stdout.contains("snapshot"));

    let event_log = fs::read_to_string(&events_path).expect("read events");
    fs::remove_file(&events_path).ok();
    assert!(event_log.contains("\"kind\":\"started\""));
    assert!(event_log.contains("plan"));
}
