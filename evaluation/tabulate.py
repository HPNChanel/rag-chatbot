"""Markdown table generation for experiment reports."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .metrics import mean_and_std


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    header_row = " | ".join(headers)
    separator = " | ".join(["---"] * len(headers))
    body = "\n".join(" | ".join(row) for row in rows)
    return f"{header_row}\n{separator}\n{body}\n"


def save_metric_table(
    metrics: Mapping[str, Sequence[float]],
    *,
    output_path: Path | str,
    baseline: str | None = None,
) -> Path:
    rows = []
    headers = ["Metric", "Mean ± Std"]
    if baseline is not None:
        headers.append("Δ vs Baseline")
    base_values = metrics.get(baseline) if baseline else None
    for name, values in metrics.items():
        mean, std = mean_and_std(values)
        row = [name, f"{mean:.3f} ± {std:.3f}"]
        if base_values is not None and name != baseline:
            base_mean, _ = mean_and_std(base_values)
            row.append(f"{mean - base_mean:+.3f}")
        elif baseline is not None:
            row.append("—")
        rows.append(row)
    table = markdown_table(headers, rows)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(table, encoding="utf-8")
    return path


__all__ = ["markdown_table", "save_metric_table"]
