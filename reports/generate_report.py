"""Render experiment reports from run artifacts."""
from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import yaml
from jinja2 import Environment, FileSystemLoader
try:  # pragma: no cover - optional dependency for charts
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - graceful fallback when matplotlib unavailable
    plt = None  # type: ignore[assignment]


@dataclass
class PlotArtifact:
    name: str
    path: str


@dataclass
class FailureCase:
    question: str
    answer: str
    context: List[str]
    categories: List[str]


@dataclass
class ErrorSummary:
    counts: Dict[str, Dict[str, float]]
    total_failures: int
    total_queries: int
    chart_path: Optional[str]
    samples: List[FailureCase]
    failures_link: Optional[str]


def _load_yaml(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _load_tables(path: Path) -> List[str]:
    tables: List[str] = []
    if not path.exists():
        return tables
    for table_path in sorted(path.glob("*.md")):
        tables.append(table_path.read_text(encoding="utf-8"))
    return tables


def _load_plots(path: Path, *, run_dir: Path) -> List[PlotArtifact]:
    if not path.exists():
        return []
    artifacts: List[PlotArtifact] = []
    for plot in sorted(path.glob("*.png")):
        try:
            relative = plot.relative_to(run_dir)
        except ValueError:
            relative = plot
        artifacts.append(PlotArtifact(plot.stem, str(relative)))
    return artifacts


def _load_failure_cases(failures_path: Path, limit: int = 5) -> List[FailureCase]:
    cases: List[FailureCase] = []
    if not failures_path.exists():
        return cases
    with failures_path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if idx >= limit:
                break
            if not line.strip():
                continue
            payload = json.loads(line)
            cases.append(
                FailureCase(
                    question=payload.get("question", ""),
                    answer=payload.get("generated_answer", ""),
                    context=[doc.get("content", "") for doc in payload.get("retrieved_docs", [])],
                    categories=list(payload.get("error_categories", [])),
                )
            )
    return cases


def _count_results(results_path: Path) -> int:
    if not results_path.exists():
        return 0
    with results_path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def _load_failure_summary(run_path: Path, total_queries: int) -> Optional[ErrorSummary]:
    failures_path = run_path / "failures.jsonl"
    if not failures_path.exists():
        return None
    counts: Dict[str, int] = {}
    payloads: List[dict] = []
    with failures_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            payloads.append(payload)
            for category in payload.get("error_categories", []):
                counts[category] = counts.get(category, 0) + 1
    if not payloads:
        return None
    chart_path = _render_failure_chart(counts, run_path)
    total_failures = len(payloads)
    summary_counts: Dict[str, Dict[str, float]] = {}
    for category, count in counts.items():
        percentage = 0.0
        if total_queries:
            percentage = (count / total_queries) * 100.0
        summary_counts[category] = {"count": count, "percentage": percentage}
    samples = _load_failure_cases(failures_path, limit=5)
    return ErrorSummary(
        counts=summary_counts,
        total_failures=total_failures,
        total_queries=total_queries,
        chart_path=chart_path,
        samples=samples,
        failures_link=str((run_path / "failures.md").relative_to(run_path)),
    )


def _render_failure_chart(counts: Dict[str, int], run_path: Path) -> Optional[str]:
    if not counts or plt is None:
        return None
    plots_dir = run_path / "plots"
    plots_dir.mkdir(exist_ok=True)
    categories = list(counts.keys())
    values = [counts[cat] for cat in categories]
    figure = plt.figure(figsize=(6, 4))
    ax = figure.add_subplot(1, 1, 1)
    positions = list(range(len(categories)))
    ax.bar(positions, values, color="#b23b3b")
    ax.set_ylabel("Count")
    ax.set_title("Failure Categories")
    ax.set_xticks(positions)
    ax.set_xticklabels(categories, rotation=45, ha="right")
    chart_path = plots_dir / "failure_categories.png"
    figure.tight_layout()
    figure.savefig(chart_path, dpi=150)
    plt.close(figure)
    return str(chart_path.relative_to(run_path))


def generate_report(
    run_dir: Path | str,
    *,
    output_format: str = "md",
    title: str = "RAG Benchmark",
    authors: Iterable[str] | None = None,
    include_appendix: bool = False,
) -> Path:
    run_path = Path(run_dir)
    config_path = run_path / "config.used.yaml"
    env_path = run_path / "env.json"
    metrics_path = run_path / "metrics.json"
    template_dir = Path(__file__).resolve().parent
    environment = Environment(loader=FileSystemLoader(str(template_dir)))
    template = environment.get_template("template.md.j2")
    config_yaml = _load_yaml(config_path)
    config = yaml.safe_load(config_yaml)
    env_payload = json.loads(env_path.read_text(encoding="utf-8")) if env_path.exists() else {}
    metrics_json = metrics_path.read_text(encoding="utf-8") if metrics_path.exists() else "{}"
    tables = _load_tables(run_path / "tables")
    plots = _load_plots(run_path / "plots", run_dir=run_path)
    total_queries = _count_results(run_path / "results_raw.jsonl")
    error_summary = _load_failure_summary(run_path, total_queries)
    failure_cases = error_summary.samples if error_summary else []
    seeds = [str(config.get("seed", 0) + idx) for idx in range(config.get("repeat", 1))]
    rendered = template.render(
        title=title,
        authors=list(authors or []),
        dataset_path=config.get("dataset_path", "unknown"),
        eval_split=config.get("eval_split", "unknown"),
        seeds=seeds,
        environment=env_payload,
        config_yaml=config_yaml,
        tables=tables,
        plots=plots,
        failure_cases=failure_cases,
        error_summary=error_summary,
        config_hash=run_path.name.split("_")[-1],
        metrics_json=metrics_json,
        include_appendix=include_appendix,
    )
    report_path = run_path / "report.md"
    report_path.write_text(rendered, encoding="utf-8")
    if output_format == "pdf":
        pandoc = shutil.which("pandoc")
        if pandoc:
            pdf_path = run_path / "report.pdf"
            subprocess.run([pandoc, str(report_path), "-o", str(pdf_path)], check=False)
            return pdf_path
    return report_path


__all__ = ["generate_report"]
