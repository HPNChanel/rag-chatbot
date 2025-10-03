import json
import shutil
from pathlib import Path

from experiments import ablations, runner
from reports.generate_report import generate_report


def _tiny_config() -> dict:
    return {
        "dataset_path": str(Path("tests/fixtures/benchmarks/tiny")),
        "eval_split": "test",
        "num_queries": 2,
        "random_query_sample": False,
        "max_docs_per_query": 2,
        "seed": 42,
        "retrieval": {
            "type": "bm25",
            "top_k": 3,
            "prf": {"enabled": False, "feedback_k": 2, "strategy": "sum"},
        },
        "reranking": {
            "enabled": False,
            "lambda_coverage": 0.1,
            "lambda_similarity": 0.7,
        },
        "generation": {
            "provider": "mock",
            "model": "echo",
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": 64,
            "citation_mode": "citation-first",
        },
        "metrics": {"retrieval_cutoffs": [1, 3]},
        "report": {
            "title": "Test",
            "authors": ["CI"],
            "output_formats": ["md"],
            "figures": {"retrieval": True},
        },
    }


def test_runner_end_to_end(tmp_path: Path):
    config = _tiny_config()
    run_dir = runner.run_experiment(config, overrides=[], repeat=1)
    try:
        metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        assert "retrieval" in metrics
        assert (run_dir / "results_raw.jsonl").exists()
    finally:
        shutil.rmtree(run_dir)


def test_ablation_suite(tmp_path: Path):
    suite_path = tmp_path / "suite.yaml"
    suite_path.write_text(
        """
base_config: experiments/configs/baseline.yaml
experiments:
  - name: one
    overrides:
      - num_queries=2
""",
        encoding="utf-8",
    )
    results = ablations.run_suite(suite_path, repeat=1, extra_overrides=["generation.provider=mock"])
    for path in results.values():
        assert (path / "metrics.json").exists()
        shutil.rmtree(path)


def test_report_generator(tmp_path: Path):
    config = _tiny_config()
    run_dir = runner.run_experiment(config, overrides=[], repeat=1)
    try:
        report_path = generate_report(run_dir, output_format="md", title="CI Report", authors=["Tester"], include_appendix=True)
        content = report_path.read_text(encoding="utf-8")
        assert "## Results" in content
        assert "Reproducibility Checklist" in content
    finally:
        shutil.rmtree(run_dir)
