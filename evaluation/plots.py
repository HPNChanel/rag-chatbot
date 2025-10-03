"""Matplotlib based plotting utilities for experiment outputs."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence

try:  # pragma: no cover - optional dependency
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except Exception:  # pragma: no cover - fallback path for CI without matplotlib
    matplotlib = None  # type: ignore
    plt = None  # type: ignore
    _HAS_MPL = False

from .metrics import mean_and_std


def _figure_path(output_dir: Path | str, name: str) -> Path:
    path = Path(output_dir) / f"{name}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def plot_bar_metrics(
    metrics: Mapping[str, Mapping[int, float]],
    *,
    title: str,
    output_dir: Path | str,
) -> Path:
    if not _HAS_MPL:
        path = _figure_path(output_dir, _safe_name(title))
        path.write_text("plot unavailable (matplotlib missing)", encoding="utf-8")
        return path
    figure = plt.figure(figsize=(8, 5))
    ax = figure.add_subplot(1, 1, 1)
    ks = sorted(next(iter(metrics.values())).keys()) if metrics else []
    width = 0.8 / max(len(metrics), 1)
    for idx, (name, values) in enumerate(metrics.items()):
        scores = [values.get(k, 0.0) for k in ks]
        ax.bar([k + idx * width for k in range(len(ks))], scores, width=width, label=name)
    ax.set_xticks([k + width * (len(metrics) - 1) / 2 for k in range(len(ks))])
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_ylabel("Score")
    ax.set_xlabel("k")
    ax.set_title(title)
    ax.legend()
    figure.tight_layout()
    path = _figure_path(output_dir, _safe_name(title))
    if _HAS_MPL:
        figure.savefig(path, dpi=200)
        plt.close(figure)
    else:  # pragma: no cover - fallback when matplotlib missing
        path.write_text("plot unavailable (matplotlib missing)", encoding="utf-8")
    return path


def plot_latency(latencies: Sequence[float], top_k_values: Sequence[int], *, output_dir: Path | str) -> Path:
    if not _HAS_MPL:
        path = _figure_path(output_dir, "latency_vs_topk")
        path.write_text("plot unavailable (matplotlib missing)", encoding="utf-8")
        return path
    figure = plt.figure(figsize=(6, 4))
    ax = figure.add_subplot(1, 1, 1)
    ax.plot(top_k_values, latencies, marker="o")
    ax.set_xlabel("top_k")
    ax.set_ylabel("Latency (s)")
    ax.set_title("Latency vs top_k")
    figure.tight_layout()
    path = _figure_path(output_dir, "latency_vs_topk")
    figure.savefig(path, dpi=200)
    plt.close(figure)
    return path


def plot_scatter(
    x: Sequence[float],
    y: Sequence[float],
    *,
    x_label: str,
    y_label: str,
    title: str,
    output_dir: Path | str,
) -> Path:
    if not _HAS_MPL:
        path = _figure_path(output_dir, _safe_name(title))
        path.write_text("plot unavailable (matplotlib missing)", encoding="utf-8")
        return path
    figure = plt.figure(figsize=(6, 4))
    ax = figure.add_subplot(1, 1, 1)
    ax.scatter(x, y)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    figure.tight_layout()
    path = _figure_path(output_dir, _safe_name(title))
    figure.savefig(path, dpi=200)
    plt.close(figure)
    return path


def _safe_name(name: str) -> str:
    slug = "".join(ch for ch in name.lower().replace(" ", "_") if ch.isalnum() or ch == "_")
    digest = hashlib.sha1(name.encode("utf-8")).hexdigest()[:8]
    return f"{slug}_{digest}"


__all__ = [
    "plot_bar_metrics",
    "plot_latency",
    "plot_scatter",
]
