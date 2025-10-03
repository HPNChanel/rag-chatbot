"""Environment capture utilities for reproducibility reporting."""
from __future__ import annotations

import importlib
import json
import platform
import sys
from pathlib import Path
from typing import Dict, Iterable

LIBRARIES_TO_CHECK: tuple[str, ...] = (
    "numpy",
    "torch",
    "faiss",
    "scipy",
    "sentence_transformers",
    "sklearn",
    "rank_bm25",
    "openai",
    "matplotlib",
)


def gather_environment_metadata() -> Dict[str, object]:
    """Collect system and library metadata for the current process."""

    metadata: Dict[str, object] = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "implementation": platform.python_implementation(),
    }
    try:
        metadata["cpu"] = platform.processor()
    except Exception:  # pragma: no cover - platform specific
        metadata["cpu"] = "unknown"
    metadata["executable"] = sys.executable
    metadata["argv"] = sys.argv
    metadata["libraries"] = {}
    for name in LIBRARIES_TO_CHECK:
        metadata["libraries"][name] = _library_version(name)
    return metadata


def _library_version(name: str) -> str | None:
    try:
        module = importlib.import_module(name)
    except Exception:
        return None
    version = getattr(module, "__version__", None)
    if version is None and hasattr(module, "VERSION"):
        version_attr = getattr(module, "VERSION")
        if isinstance(version_attr, tuple):
            version = ".".join(str(part) for part in version_attr)
        else:
            version = str(version_attr)
    return version


def write_environment_snapshot(path: Path | str) -> Dict[str, object]:
    """Write environment metadata to ``path`` and return the captured dictionary."""

    metadata = gather_environment_metadata()
    destination = Path(path)
    destination.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return metadata


__all__ = ["gather_environment_metadata", "write_environment_snapshot"]
