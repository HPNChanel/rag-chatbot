"""Evaluation utilities for text generation quality."""
from __future__ import annotations

from typing import Iterable, List

from nltk.translate.bleu_score import corpus_bleu
from rouge_score import rouge_scorer


def bleu_score(references: Iterable[List[str]], hypotheses: Iterable[str]) -> float:
    reference_list = [[ref.split()] for ref in references]
    hypothesis_tokens = [hyp.split() for hyp in hypotheses]
    return corpus_bleu(reference_list, hypothesis_tokens)


def rouge_scores(references: Iterable[str], hypotheses: Iterable[str]) -> dict:
    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    results = {"rouge1": [], "rougeL": []}
    for ref, hyp in zip(references, hypotheses):
        score = scorer.score(ref, hyp)
        results["rouge1"].append(score["rouge1"].fmeasure)
        results["rougeL"].append(score["rougeL"].fmeasure)
    return {metric: sum(values) / len(values) if values else 0.0 for metric, values in results.items()}


def human_eval_placeholder() -> str:
    """Placeholder for manual evaluation steps."""

    return (
        "Run a structured human evaluation with domain experts. "
        "Capture ratings for helpfulness, factuality, and grounding."
    )
