"""Statistical significance utilities for comparing experiment runs."""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence


@dataclass
class BootstrapResult:
    p_value: float
    ci_low: float
    ci_high: float


def paired_bootstrap(
    baseline: Sequence[float],
    contender: Sequence[float],
    *,
    n_boot: int = 1000,
    metric_name: str | None = None,
    seed: int | None = None,
) -> BootstrapResult:
    if len(baseline) != len(contender):
        raise ValueError("Baseline and contender must have the same number of samples")
    if not baseline:
        raise ValueError("At least one score is required for bootstrap")
    rng = random.Random(seed)
    diffs = [b - c for b, c in zip(contender, baseline)]
    observed = sum(diffs) / len(diffs)
    samples = []
    more_extreme = 0
    for _ in range(n_boot):
        resampled = [diffs[rng.randrange(len(diffs))] for _ in diffs]
        mean_diff = sum(resampled) / len(resampled)
        samples.append(mean_diff)
        if abs(mean_diff) >= abs(observed):
            more_extreme += 1
    alpha = 0.05
    lower_idx = int(alpha / 2 * n_boot)
    upper_idx = int((1 - alpha / 2) * n_boot)
    samples.sort()
    ci_low = samples[max(lower_idx - 1, 0)]
    ci_high = samples[min(upper_idx, n_boot - 1)]
    p_value = (more_extreme + 1) / (n_boot + 1)
    return BootstrapResult(p_value=p_value, ci_low=ci_low, ci_high=ci_high)


__all__ = ["BootstrapResult", "paired_bootstrap"]
