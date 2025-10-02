"""Evaluation utilities for retrieval components."""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Sequence, Set

from data_ingestion.loader import Document


def precision_at_k(relevant: Set[str], retrieved: Sequence[str], k: int) -> float:
    if k <= 0:
        return 0.0
    retrieved_at_k = retrieved[:k]
    hits = sum(1 for doc_id in retrieved_at_k if doc_id in relevant)
    return hits / min(k, len(retrieved_at_k)) if retrieved_at_k else 0.0


def recall_at_k(relevant: Set[str], retrieved: Sequence[str], k: int) -> float:
    if not relevant:
        return 0.0
    retrieved_at_k = retrieved[:k]
    hits = sum(1 for doc_id in retrieved_at_k if doc_id in relevant)
    return hits / len(relevant)


def coverage(relevant: Set[str], retrieved: Sequence[str]) -> float:
    if not relevant:
        return 0.0
    hits = sum(1 for doc_id in retrieved if doc_id in relevant)
    return hits / len(relevant)


def evaluate_retrieval(
    queries: Dict[str, Set[str]],
    retrieval_results: Dict[str, List[str]],
    k_values: Sequence[int] = (1, 3, 5),
) -> Dict[str, Dict[int, float]]:
    """Return aggregate retrieval metrics for multiple queries."""

    metrics: Dict[str, Dict[int, float]] = {"precision": {}, "recall": {}}
    coverage_scores: List[float] = []
    for k in k_values:
        precisions: List[float] = []
        recalls: List[float] = []
        for query_id, relevant_docs in queries.items():
            retrieved_docs = retrieval_results.get(query_id, [])
            precisions.append(precision_at_k(relevant_docs, retrieved_docs, k))
            recalls.append(recall_at_k(relevant_docs, retrieved_docs, k))
            coverage_scores.append(coverage(relevant_docs, retrieved_docs))
        metrics["precision"][k] = sum(precisions) / len(precisions) if precisions else 0.0
        metrics["recall"][k] = sum(recalls) / len(recalls) if recalls else 0.0
    metrics["coverage"] = {0: sum(coverage_scores) / len(coverage_scores) if coverage_scores else 0.0}
    return metrics
