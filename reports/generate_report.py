"""Render experiment reports from run artifacts."""
from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import yaml
from jinja2 import Environment, FileSystemLoader


@dataclass
class PlotArtifact:
    name: str
    path: str


@dataclass
class FailureCase:
    question: str
    answer: str
    context: List[str]


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


def _load_failure_cases(results_path: Path, limit: int = 5) -> List[FailureCase]:
    cases: List[FailureCase] = []
    if not results_path.exists():
        return cases
    for idx, line in enumerate(results_path.read_text(encoding="utf-8").splitlines()):
        if idx >= limit:
            break
        payload = json.loads(line)
        cases.append(
            FailureCase(
                question=payload.get("question", ""),
                answer=payload.get("generated_answer", ""),
                context=payload.get("selected_context", []),
            )
        )
    return cases


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
    failure_cases = _load_failure_cases(run_path / "results_raw.jsonl")
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
