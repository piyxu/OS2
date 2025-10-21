use std::fs::{self, File};
use std::io::Write;
use std::path::PathBuf;
use std::process::Command;

use os2_kernel::{EventKind, KernelEvent, TokenId};

fn write_temp_log() -> PathBuf {
    let mut path = std::env::temp_dir();
    let unique = format!("os2_event_log_{}.jsonl", std::process::id());
    path.push(unique);

    let mut file = File::create(&path).expect("create log file");

    let events = vec![
        KernelEvent {
            token_id: TokenId::new(1),
            kind: EventKind::Started,
            detail: serde_json::Value::Null,
            timestamp: 1,
        }
        .to_json(),
        KernelEvent {
            token_id: TokenId::new(1),
            kind: EventKind::Custom("demo".into()),
            detail: serde_json::json!({"ok": true}),
            timestamp: 2,
        }
        .to_json(),
        KernelEvent {
            token_id: TokenId::new(1),
            kind: EventKind::Completed,
            detail: serde_json::Value::Null,
            timestamp: 3,
        }
        .to_json(),
    ];

    for event in events {
        let line = serde_json::to_string(&event).expect("serialize event");
        writeln!(file, "{}", line).expect("write event line");
    }

    path
}

#[test]
fn replay_cli_outputs_summary() {
    let path = write_temp_log();

    let exe = env!("CARGO_BIN_EXE_event_replay");
    let output = Command::new(exe)
        .arg(&path)
        .output()
        .expect("run event_replay");

    fs::remove_file(&path).ok();

    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).expect("stdout utf8");
    assert!(stdout.contains("token 1 started"));
    assert!(stdout.contains("demo event"));
    assert!(stdout.contains("Summary: 3 events"));
    assert!(stdout.contains("Metrics: total_tokens=1 completed=1 failed=0 win_rate=100.0%"));
    assert!(stdout.contains("average_latency=2.00 ticks"));
}
