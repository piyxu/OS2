from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest

from cli.ai_determinism import configure_deterministic_mode


@pytest.fixture()
def stub_modules(monkeypatch):
    numpy_calls = {}

    def numpy_seed(value):
        numpy_calls["seed"] = value

    numpy_module = SimpleNamespace(random=SimpleNamespace(seed=numpy_seed))

    torch_calls = {}

    class DummyCuda:
        def __init__(self) -> None:
            self.seed = None

        def is_available(self) -> bool:
            return True

        def manual_seed_all(self, value: int) -> None:
            torch_calls["cuda_seed"] = value

    def manual_seed(value: int) -> None:
        torch_calls["seed"] = value

    def deterministic(flag: bool) -> None:
        torch_calls["deterministic"] = flag

    torch_module = SimpleNamespace(
        manual_seed=manual_seed,
        use_deterministic_algorithms=deterministic,
        cuda=DummyCuda(),
    )

    transformers_calls = {}

    def set_seed(value: int) -> None:
        transformers_calls["seed"] = value

    transformers_module = SimpleNamespace(set_seed=set_seed)

    monkeypatch.setitem(sys.modules, "numpy", numpy_module)
    monkeypatch.setitem(sys.modules, "torch", torch_module)
    monkeypatch.setitem(sys.modules, "transformers", transformers_module)

    yield numpy_calls, torch_calls, transformers_calls

    for name in ("numpy", "torch", "transformers"):
        sys.modules.pop(name, None)


def test_configure_deterministic_mode_sets_framework_seeds(stub_modules):
    numpy_calls, torch_calls, transformers_calls = stub_modules
    metadata = configure_deterministic_mode("sample-seed-material")

    assert metadata["seed"] == int(metadata["seed_digest"][:16], 16)
    assert metadata["environment"]["PYTHONHASHSEED"] == str(metadata["random_seed"])
    assert os.environ["TRANSFORMERS_OFFLINE"] == "1"
    assert numpy_calls["seed"] == metadata["random_seed"]
    assert torch_calls["seed"] == metadata["random_seed"]
    assert torch_calls["cuda_seed"] == metadata["random_seed"]
    assert torch_calls["deterministic"] is True
    assert transformers_calls["seed"] == metadata["random_seed"]
    assert metadata["frameworks"]["torch"]["cuda"] is True
