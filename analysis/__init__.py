"""High level orchestration helpers for analysis workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Mapping

import yaml

from evaluation.datasets import load_dataset
from reports.generate_report import generate_report

from .errors import ErrorAnalyzer
from .redaction import RedactionSettings, TextRedactor
from .viz_failures import export_failure_visualisations


def run_failure_analysis(
    run_dir: Path | str,
    *,
    enable_redaction: bool | None = None,
    redact_pii: bool | None = None,
    sensitive_terms: Iterable[str] | None = None,
    update_report: bool = True,
) -> List[Dict[str, object]]:
    """Run error analysis for an experiment directory."""

    run_path = Path(run_dir)
    results_path = run_path / "results_raw.jsonl"
    if not results_path.exists():
        raise FileNotFoundError(f"Expected results at {results_path}")
    config_path = run_path / "config.used.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Expected config at {config_path}")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    dataset = load_dataset(config.get("dataset_path", "tiny"))
    gold_labels = _build_gold_labels(dataset)
    redaction_config = (
        config.get("safety", {}) if isinstance(config.get("safety"), Mapping) else {}
    )
    settings = RedactionSettings(
        enable_redaction=(
            enable_redaction
            if enable_redaction is not None
            else bool(redaction_config.get("enable_redaction", False))
        ),
        redact_pii=(
            redact_pii
            if redact_pii is not None
            else bool(redaction_config.get("pii", True))
        ),
        sensitive_terms=list(
            sensitive_terms or redaction_config.get("sensitive_terms", [])
        ),
    )
    redactor = TextRedactor(settings)
    doc_lookup = {record.id: record.text for record in dataset.corpus}
    analyzer = ErrorAnalyzer(run_path, redactor=redactor, doc_lookup=doc_lookup)
    gold_payload = _enrich_gold_labels(gold_labels, doc_lookup)
    failures = analyzer.extract_failure_cases(results_path, gold_payload)
    export_failure_visualisations(failures, run_path)
    if update_report:
        report_cfg = (
            config.get("report", {})
            if isinstance(config.get("report"), Mapping)
            else {}
        )
        generate_report(
            run_path,
            title=report_cfg.get("title", "RAG Benchmark"),
            authors=report_cfg.get("authors", []),
            include_appendix=bool(report_cfg.get("include_appendix", False)),
        )
    return failures


def _build_gold_labels(dataset) -> Dict[str, Dict[str, object]]:
    labels: Dict[str, Dict[str, object]] = {}
    for query in dataset.queries:
        labels[query.id] = {
            "question": query.question,
            "gold_answers": list(query.gold_answers),
            "gold_doc_ids": list(query.gold_passages),
        }
    return labels


def _enrich_gold_labels(
    gold_labels: Mapping[str, Dict[str, object]],
    doc_lookup: Mapping[str, str],
) -> Dict[str, Dict[str, object]]:
    enriched: Dict[str, Dict[str, object]] = {}
    for query_id, payload in gold_labels.items():
        enriched[query_id] = dict(payload)
        enriched[query_id]["doc_lookup"] = doc_lookup
    return enriched


__all__ = ["run_failure_analysis"]
