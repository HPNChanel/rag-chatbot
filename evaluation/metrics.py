"""Evaluation metrics for retrieval and generation."""
from __future__ import annotations

import math
import statistics
from collections import Counter
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

from rouge_score import rouge_scorer

try:  # pragma: no cover - optional dependency
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
    from nltk.translate.meteor_score import meteor_score
except Exception:  # pragma: no cover
    SmoothingFunction = None  # type: ignore
    sentence_bleu = None  # type: ignore
    meteor_score = None  # type: ignore


def precision_at_k(relevant: Sequence[str], retrieved: Sequence[str], k: int) -> float:
    if k <= 0:
        return 0.0
    retrieved_at_k = retrieved[:k]
    if not retrieved_at_k:
        return 0.0
    hits = sum(1 for doc_id in retrieved_at_k if doc_id in relevant)
    return hits / min(k, len(retrieved_at_k))


def recall_at_k(relevant: Sequence[str], retrieved: Sequence[str], k: int) -> float:
    if not relevant:
        return 0.0
    retrieved_at_k = retrieved[:k]
    hits = sum(1 for doc_id in retrieved_at_k if doc_id in relevant)
    return hits / len(relevant)


def mean_reciprocal_rank(relevant: Sequence[str], retrieved: Sequence[str]) -> float:
    for idx, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            return 1.0 / idx
    return 0.0


def ndcg_at_k(relevant: Sequence[str], retrieved: Sequence[str], k: int) -> float:
    if k <= 0:
        return 0.0
    denom = math.log2
    gains = [1.0 if doc in relevant else 0.0 for doc in retrieved[:k]]
    if not gains:
        return 0.0
    dcg = sum(gain / denom(idx + 1) for idx, gain in enumerate(gains, start=1))
    ideal = sorted(gains, reverse=True)
    idcg = sum(gain / denom(idx + 1) for idx, gain in enumerate(ideal, start=1))
    return dcg / idcg if idcg > 0 else 0.0


def coverage_score(retrieved: Sequence[str]) -> float:
    if not retrieved:
        return 0.0
    sources = {doc_id.split("::")[0] for doc_id in retrieved}
    return len(sources) / len(retrieved)


def aggregate_retrieval_metrics(
    judgements: Mapping[str, Sequence[str]],
    results: Mapping[str, Sequence[str]],
    *,
    k_values: Sequence[int],
) -> Dict[str, Dict[int, float]]:
    metrics: Dict[str, Dict[int, float]] = {
        "precision": {},
        "recall": {},
        "ndcg": {},
    }
    mrr_scores: List[float] = []
    coverage_scores: List[float] = []
    for query_id, relevant in judgements.items():
        retrieved = list(results.get(query_id, []))
        mrr_scores.append(mean_reciprocal_rank(relevant, retrieved))
        for k in k_values:
            precision_scores = metrics["precision"].setdefault(k, [])  # type: ignore[assignment]
            recall_scores = metrics["recall"].setdefault(k, [])  # type: ignore[assignment]
            ndcg_scores = metrics["ndcg"].setdefault(k, [])  # type: ignore[assignment]
            precision_scores.append(precision_at_k(relevant, retrieved, k))
            recall_scores.append(recall_at_k(relevant, retrieved, k))
            ndcg_scores.append(ndcg_at_k(relevant, retrieved, k))
            coverage_scores.append(coverage_score(retrieved[:k]))
    for k in k_values:
        precision_scores = metrics["precision"][k]
        recall_scores = metrics["recall"][k]
        ndcg_scores = metrics["ndcg"][k]
        metrics["precision"][k] = statistics.fmean(precision_scores) if precision_scores else 0.0
        metrics["recall"][k] = statistics.fmean(recall_scores) if recall_scores else 0.0
        metrics["ndcg"][k] = statistics.fmean(ndcg_scores) if ndcg_scores else 0.0
    metrics["coverage"] = {0: statistics.fmean(coverage_scores) if coverage_scores else 0.0}
    metrics["mrr"] = {0: statistics.fmean(mrr_scores) if mrr_scores else 0.0}
    return metrics


def rouge_l(reference: str, prediction: str) -> float:
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    result = scorer.score(reference, prediction)
    return float(result["rougeL"].fmeasure)


def bleu(reference: str, prediction: str) -> float:
    if sentence_bleu is None or SmoothingFunction is None:  # pragma: no cover
        return _overlap_ratio(reference, prediction)
    smoothing = SmoothingFunction().method3
    try:
        return float(
            sentence_bleu(
                [reference.split()],
                prediction.split(),
                smoothing_function=smoothing,
            )
        )
    except Exception:  # pragma: no cover - fallback for incompatible nltk versions
        return _overlap_ratio(reference, prediction)


def meteor(reference: str, prediction: str) -> float:
    if meteor_score is None:  # pragma: no cover
        return _overlap_ratio(reference, prediction)
    try:
        return float(meteor_score([reference], prediction))
    except (LookupError, TypeError):  # pragma: no cover - missing data or version mismatch
        return _overlap_ratio(reference, prediction)


def _overlap_ratio(reference: str, prediction: str) -> float:
    ref_tokens = reference.lower().split()
    pred_tokens = prediction.lower().split()
    if not ref_tokens or not pred_tokens:
        return 0.0
    ref_counts = Counter(ref_tokens)
    pred_counts = Counter(pred_tokens)
    overlap = sum(min(ref_counts[token], pred_counts[token]) for token in pred_counts)
    return overlap / max(len(pred_tokens), 1)


def aggregate_generation_metrics(
    references: Mapping[str, Sequence[str]],
    predictions: Mapping[str, str],
) -> Dict[str, float]:
    rouge_scores: List[float] = []
    bleu_scores: List[float] = []
    meteor_scores: List[float] = []
    for query_id, refs in references.items():
        prediction = predictions.get(query_id, "")
        if not refs:
            continue
        rouge_scores.append(max(rouge_l(ref, prediction) for ref in refs))
        bleu_scores.append(max(bleu(ref, prediction) for ref in refs))
        meteor_scores.append(max(meteor(ref, prediction) for ref in refs))
    return {
        "rougeL": statistics.fmean(rouge_scores) if rouge_scores else 0.0,
        "BLEU": statistics.fmean(bleu_scores) if bleu_scores else 0.0,
        "METEOR": statistics.fmean(meteor_scores) if meteor_scores else 0.0,
    }


def mean_and_std(values: Sequence[float]) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return values[0], 0.0
    return statistics.fmean(values), statistics.pstdev(values)


__all__ = [
    "aggregate_generation_metrics",
    "aggregate_retrieval_metrics",
    "bleu",
    "coverage_score",
    "mean_and_std",
    "mean_reciprocal_rank",
    "meteor",
    "ndcg_at_k",
    "precision_at_k",
    "recall_at_k",
    "rouge_l",
]
