"""Convenience runners for standard ablation suites."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import yaml

from . import runner


class AblationSuiteError(RuntimeError):
    pass


def run_suite(path: Path | str, *, repeat: int = 1, extra_overrides: Sequence[str] | None = None) -> Dict[str, Path]:
    suite_path = Path(path)
    if not suite_path.exists():
        raise FileNotFoundError(f"Ablation suite not found: {suite_path}")
    payload = yaml.safe_load(suite_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise AblationSuiteError("Suite file must be a mapping")
    base_config_path = Path(payload.get("base_config"))
    if not base_config_path.exists():
        raise AblationSuiteError(f"Base config missing: {base_config_path}")
    config = runner._load_config(base_config_path)  # type: ignore[attr-defined]
    results: Dict[str, Path] = {}
    experiments = payload.get("experiments", [])
    if not experiments:
        raise AblationSuiteError("No experiments defined in suite")
    for experiment in experiments:
        name = experiment.get("name")
        overrides = list(experiment.get("overrides", []))
        if extra_overrides:
            overrides.extend(extra_overrides)
        if not name:
            raise AblationSuiteError("Experiment entries must define a name")
        results[name] = runner.run_experiment(config, overrides=overrides, repeat=repeat)
    return results


__all__ = ["run_suite", "AblationSuiteError"]
