⚠️ Notice: The installation guide and deterministic system currently contain errors. Please continue using the current version until these issues are fixed.
# OS2 – A Microkernel Playground for AI Agents

OS2 is a hobby project that acts as a microkernel-inspired operating system within your existing environment. Instead of trying to replace your desktop OS, it provides a safe playground where AI agents can experiment with deterministic scheduling, snapshots, and capability-governed tools without risking the host machine. If enough curious builders gather around the idea, we’ll keep evolving it together and build a shared community space.

The project was initially prototyped and refined with the assistance of Codex, whose tooling helped accelerate early development.

## Why you might fall in love with it
- **Deterministic AI kernel.** Every task is logged through a hash-chained ledger, so you can replay reasoning tokens, Python sessions, and git/pip interactions exactly as they happened.
- **Python-first shell.** `python`, `pip`, `pyvm`, and `pyx` run inside an audited VM that now streams output, honours token budgets, and mirrors transcripts into snapshot metadata.
- **Rust microkernel core.** The `os2-kernel` crate schedules capability-scoped tokens, records checkpoints, and exposes WASM sidecars for future extensions.
- **Observability baked in.** Snapshots, metrics, and ledger events form an audit trail that explains why the system behaved the way it did.
- **Experiment-friendly design.** The shell exposes helpers for cloning repos, installing dependencies, and replaying logs—perfect for tinkering agents.

## Honest limitations (for now)
- **Linux bias.** Most tooling assumes a Linux host with the Rust toolchain installed; Windows and macOS builds are currently untested.
- **Single-node focus.** Federation hooks exist, but distributed roll-outs still need polish before they can run unsupervised.
- **Manual provisioning.** You create environments and manage secrets yourself—there is no automated cloud bootstrap yet.
- **Curated documentation.** This README plus the consolidated technical overview (`docs/system_overview.md`) and journey log (`docs/project_journey.md`) cover architecture, kernel protocol, shell commands, and release practices without redundant copies.
- **Community in progress.** We are looking for the first wave of collaborators—drop a note if you want to join the adventure.

## Getting started

### 1. Install base packages (Debian/Ubuntu example)
```bash
sudo apt-get update && sudo apt-get install \
    build-essential \
    python3-venv \
    ninja-build \
    cmake
```

### 2. Install the Rust toolchain
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"
rustup default stable
rustup update
```
Confirm your setup with `rustc --version` and `cargo --version`.

### 3. Prepare the Python environment
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Build the kernel and run the shell
```bash
make build-kernel
make run-python-host
python -m build
python3 cli/command_shell.py
```
### 5. Explore the shell essentials
```bash
help                             # list available shell commands and usage hints
python sample.py                 # run workspace scripts with streaming output
pip install torch                # full pip client routed through the sandbox
git clone https://github.com/... # safe git wrapper with live progress
snapshot-auth 5                  # mint an authenticated snapshot token
python --token-budget 800 app.py # raise per-run token limits when needed
```
All commands stream to the console, capture transcripts for replay, and record their footprints in the snapshot ledger.

## Repository map
- `cli/` – Deterministic command shell, Python VM launcher, and regression tests.
- `docs/` – [system overview](docs/system_overview.md) and the English [project journey](docs/project_journey.md).
- `rust/os2-kernel/` – Rust microkernel crate with examples and scheduler logic.
- `scripts/` – automation snippets and sample shell sessions.

## Join the experiment
The documentation and code may still contain rough edges or mistakes — this project moves fast and evolves often.
If you encounter any issues or have questions, don’t hesitate to reach out or open a discussion. Together, we can refine and evolve OS2 into something truly remarkable.

## License
OS2 ships under the GPLv3 license. Third-party assets keep their original terms.
