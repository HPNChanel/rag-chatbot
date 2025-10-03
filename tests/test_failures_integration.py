from __future__ import annotations

import json
from pathlib import Path

import yaml
from typer.testing import CliRunner

from experiments.cli import app


def _write_run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "20240101_test"
    run_dir.mkdir(parents=True)
    config = {
        "dataset_path": str(Path("tests/fixtures/benchmarks/tiny").resolve()),
        "eval_split": "test",
        "seed": 123,
        "safety": {
            "enable_redaction": False,
            "pii": True,
            "sensitive_terms": ["Neptune"],
        },
    }
    (run_dir / "config.used.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    failure_payload = {
        "query_id": "q1",
        "question": "Which planet is called the red planet?",
        "retrieved": [{"doc_id": "doc1", "score": 1.0}],
        "selected_context": ["doc1"],
        "generated_answer": "Email me at jane.doe@example.com. Neptune is the red planet. [doc99]",
        "citations": ["doc99"],
    }
    (run_dir / "results_raw.jsonl").write_text(
        json.dumps(failure_payload) + "\n", encoding="utf-8"
    )
    return run_dir


def test_cli_generates_redacted_failures(tmp_path):
    run_dir = _write_run(tmp_path)
    runner = CliRunner()
    result = runner.invoke(app, ["analyze", str(run_dir), "--redact"])
    assert result.exit_code == 0, result.output
    failures_path = run_dir / "failures.jsonl"
    assert failures_path.exists()
    with failures_path.open("r", encoding="utf-8") as handle:
        entries = [json.loads(line) for line in handle if line.strip()]
    assert entries, "No failures recorded"
    failure = entries[0]
    assert "[REDACTED_EMAIL]" in failure["generated_answer"]
    assert "[REDACTED_TERM]" in failure["generated_answer"]
    assert "Citation Error" in failure["error_categories"]
    assert (run_dir / "failures.md").exists()
    markdown = (run_dir / "failures.md").read_text(encoding="utf-8")
    assert "[REDACTED_EMAIL]" in markdown
    report = (run_dir / "report.md").read_text(encoding="utf-8")
    assert "## Error Analysis" in report
    assert "Citation Error" in report
    assert "[REDACTED_EMAIL]" in report
    html_report = run_dir / "failures.html"
    assert html_report.exists()
