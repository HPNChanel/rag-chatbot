"""Utilities for deterministic experiment execution."""
from __future__ import annotations

import os
import random
from typing import Any

try:  # pragma: no cover - optional dependency
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore

try:  # pragma: no cover - optional dependency
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore

try:  # pragma: no cover - optional dependency
    import faiss
except Exception:  # pragma: no cover
    faiss = None  # type: ignore


def _set_env_seed(seed: int) -> None:
    os.environ.setdefault("PYTHONHASHSEED", str(seed))


def fix_seeds(seed: int) -> None:
    """Set seeds across common libraries used by the project."""

    if seed < 0:
        raise ValueError("Seed must be non-negative")
    _set_env_seed(seed)
    random.seed(seed)
    if np is not None:
        np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
    if faiss is not None:
        if hasattr(faiss, "randu64"):
            faiss.randu64(0, seed)  # type: ignore[attr-defined]
        if hasattr(faiss, "random_seed"):
            faiss.random_seed(seed)  # type: ignore[attr-defined]


__all__ = ["fix_seeds"]
