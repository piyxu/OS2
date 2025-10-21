# PIYXU OS2 0.1.0 version build orchestration Makefile

# Python is the sole host environment. Default to Linux-style toolchains while
# allowing callers to override the cargo target if required for local testing.
PYTHON ?= python3
DEFAULT_CARGO_TARGET ?= x86_64-unknown-linux-gnu

KERNEL_MANIFEST := rust/os2-kernel/Cargo.toml
CARGO_TARGET ?= $(DEFAULT_CARGO_TARGET)
KERNEL_BUILD_DIR ?= target/$(CARGO_TARGET)/release
PYTHONPATH ?= $(CURDIR)
SMOKE_SCRIPT ?= scripts/smoke_test.os2
CARGO_BIN := cargo
RUST_TOOLCHAIN ?= nightly

BAREMETAL_MANIFEST := rust/os2-boot/Cargo.toml
BAREMETAL_TARGET_SPEC := rust/os2-boot/x86_64-bootloader.json
BAREMETAL_TARGET := x86_64-unknown-none
BAREMETAL_BUILD_DIR := rust/os2-boot/target/$(BAREMETAL_TARGET)/release

# Flags shared across cargo invocations.
CARGO_BUILD_FLAGS := --release --manifest-path $(KERNEL_MANIFEST) --bin kernel_daemon --target $(CARGO_TARGET)

.PHONY: all build-kernel build-kernel-baremetal run-python-host clean

all: build-kernel

build-kernel:
	@command -v $(CARGO_BIN) >/dev/null 2>&1 || { echo "error: cargo not found in PATH" >&2; exit 1; }
	$(CARGO_BIN) build $(CARGO_BUILD_FLAGS)

build-kernel-baremetal:
	@command -v $(CARGO_BIN) >/dev/null 2>&1 || { echo "error: cargo not found in PATH" >&2; exit 1; }
	$(CARGO_BIN) +$(RUST_TOOLCHAIN) build \
		-Zbuild-std=core,alloc,compiler_builtins \
		-Zbuild-std-features=compiler-builtins-mem \
		--release \
		--manifest-path $(BAREMETAL_MANIFEST) \
		--target $(BAREMETAL_TARGET_SPEC)
	@echo "Bare-metal image available under $(BAREMETAL_BUILD_DIR)"

run-python-host: build-kernel
	@command -v $(PYTHON) >/dev/null 2>&1 || { echo "error: $(PYTHON) not found in PATH" >&2; exit 1; }
	KERNEL_BUILD_DIR=$(KERNEL_BUILD_DIR) PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m cli.command_shell --script $(SMOKE_SCRIPT)

clean:
	@if command -v $(CARGO_BIN) >/dev/null 2>&1; then \
		$(CARGO_BIN) clean --manifest-path $(KERNEL_MANIFEST); \
		$(CARGO_BIN) clean --manifest-path $(BAREMETAL_MANIFEST); \
	else echo "warning: cargo not found, skipping Rust clean"; fi
	@$(PYTHON) -c "import pathlib, shutil; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__')]; [py.unlink(missing_ok=True) for py in pathlib.Path('.').rglob('*.pyc')]"
