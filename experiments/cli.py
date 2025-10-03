"""Command line entry points for the benchmarking suite."""
from __future__ import annotations

from pathlib import Path
from typing import List

import typer

from analysis import run_failure_analysis

from . import ablations, runner
from reports.generate_report import generate_report

app = typer.Typer(add_completion=False)


def _run(
    config: Path = typer.Option(..., exists=True, dir_okay=False, help="Path to experiment YAML config"),
    override: List[str] = typer.Option([], help="Override config values"),
    repeat: int = typer.Option(1, help="Number of seeds to run"),
) -> None:
    payload = runner._load_config(config)  # type: ignore[attr-defined]
    run_dir = runner.run_experiment(payload, overrides=override, repeat=repeat)
    typer.echo(str(run_dir))


@app.command()
def run(
    config: Path = typer.Argument(..., exists=True, dir_okay=False),
    repeat: int = typer.Option(1, help="Number of seeds to run"),
    override: List[str] = typer.Option([], help="Override config values"),
) -> None:
    payload = runner._load_config(config)  # type: ignore[attr-defined]
    run_dir = runner.run_experiment(payload, overrides=override, repeat=repeat)
    typer.echo(str(run_dir))


@app.command()
def ablate(
    suite: Path = typer.Argument(..., exists=True, dir_okay=False),
    repeat: int = typer.Option(1, help="Number of seeds per experiment"),
    override: List[str] = typer.Option([], help="Global overrides applied to all experiments"),
) -> None:
    results = ablations.run_suite(suite, repeat=repeat, extra_overrides=override)
    for name, path in results.items():
        typer.echo(f"{name}: {path}")


@app.command()
def report(
    run_dir: Path = typer.Argument(..., exists=True, file_okay=False),
    format: str = typer.Option("md", "--format", help="Output format"),
    title: str = typer.Option("Benchmark Report", help="Report title"),
    authors: List[str] = typer.Option([], help="Report authors"),
    appendix: bool = typer.Option(False, help="Include appendix"),
) -> None:
    output = generate_report(run_dir, output_format=format, title=title, authors=authors, include_appendix=appendix)
    typer.echo(str(output))


@app.command()
def analyze(
    run_dir: Path = typer.Argument(..., exists=True, file_okay=False),
    redact: bool = typer.Option(False, "--redact", help="Enable safety redaction"),
) -> None:
    run_failure_analysis(run_dir, enable_redaction=redact if redact else None)
    typer.echo(str(run_dir / "failures.jsonl"))


def run_command() -> None:
    typer.run(run)


def ablations_command() -> None:
    typer.run(ablate)


def report_command() -> None:
    typer.run(report)


def analyze_command() -> None:
    typer.run(analyze)


__all__ = ["run_command", "ablations_command", "report_command", "analyze_command"]
