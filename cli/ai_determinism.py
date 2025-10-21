"""Helpers for enforcing deterministic behaviour across AI frameworks."""

from __future__ import annotations

import hashlib
import os
import random
from typing import Any, Dict


def _derive_seed(seed_material: str) -> Dict[str, int | str]:
    digest = hashlib.sha256(seed_material.encode("utf-8")).hexdigest()
    seed = int(digest[:16], 16)
    random_seed = seed % (2**32)
    return {"seed": seed, "random_seed": random_seed, "digest": digest}


def configure_deterministic_mode(seed_material: str) -> Dict[str, Any]:
    """Configure deterministic execution across supported frameworks.

    The function returns a JSON-serialisable dictionary describing the
    deterministic configuration that was applied.  The metadata includes the
    derived seed material, environment variables that were set, and framework
    specific status blocks indicating whether deterministic configuration
    succeeded for each optional dependency.
    """

    seed_info = _derive_seed(seed_material)
    random_seed = int(seed_info["random_seed"])
    random.seed(random_seed)

    os.environ["PYTHONHASHSEED"] = str(random_seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    metadata: Dict[str, Any] = {
        "seed": int(seed_info["seed"]),
        "seed_digest": seed_info["digest"],
        "random_seed": random_seed,
        "environment": {
            "PYTHONHASHSEED": os.environ["PYTHONHASHSEED"],
            "CUBLAS_WORKSPACE_CONFIG": os.environ["CUBLAS_WORKSPACE_CONFIG"],
            "TRANSFORMERS_OFFLINE": os.environ["TRANSFORMERS_OFFLINE"],
        },
        "frameworks": {
            "python": {"random_seed": random_seed},
        },
    }

    # Optional numpy support -------------------------------------------------
    try:  # pragma: no branch - executed when numpy is present
        import numpy as np  # type: ignore

        np.random.seed(random_seed)
        metadata["frameworks"]["numpy"] = {"available": True, "seed": random_seed}
    except ModuleNotFoundError:
        metadata["frameworks"]["numpy"] = {"available": False}
    except Exception as exc:  # pragma: no cover - defensive
        metadata["frameworks"]["numpy"] = {"available": True, "error": str(exc)[:200]}

    # Optional torch support -------------------------------------------------
    try:  # pragma: no branch - executed when torch is present
        import torch  # type: ignore

        torch.manual_seed(random_seed)
        cuda_available = False
        if hasattr(torch, "cuda") and callable(getattr(torch.cuda, "is_available", None)):
            cuda_available = bool(torch.cuda.is_available())
            if cuda_available and hasattr(torch.cuda, "manual_seed_all"):
                torch.cuda.manual_seed_all(random_seed)
        if hasattr(torch, "use_deterministic_algorithms"):
            torch.use_deterministic_algorithms(True)
        metadata["frameworks"]["torch"] = {
            "available": True,
            "seed": random_seed,
            "cuda": cuda_available,
            "deterministic": True,
        }
    except ModuleNotFoundError:
        metadata["frameworks"]["torch"] = {"available": False}
    except Exception as exc:  # pragma: no cover - defensive
        metadata["frameworks"]["torch"] = {"available": True, "error": str(exc)[:200]}

    # Optional transformers support -----------------------------------------
    try:  # pragma: no branch - executed when transformers is present
        import transformers  # type: ignore

        if hasattr(transformers, "set_seed"):
            transformers.set_seed(random_seed)
        metadata["frameworks"]["transformers"] = {
            "available": True,
            "seed": random_seed,
            "offline": os.environ["TRANSFORMERS_OFFLINE"] == "1",
        }
    except ModuleNotFoundError:
        metadata["frameworks"]["transformers"] = {"available": False}
    except Exception as exc:  # pragma: no cover - defensive
        metadata["frameworks"]["transformers"] = {
            "available": True,
            "error": str(exc)[:200],
        }

    return metadata


__all__ = ["configure_deterministic_mode"]
