"""Evaluation utilities for retrieval components."""
from __future__ import annotations

from typing import Dict, List, Mapping, Sequence, Set


def precision_at_k(relevant: Set[str], retrieved: Sequence[str], k: int) -> float:
    """Return precision@k for a single query."""

    if k <= 0:
        return 0.0
    retrieved_at_k = retrieved[:k]
    if not retrieved_at_k:
        return 0.0
    hits = sum(1 for doc_id in retrieved_at_k if doc_id in relevant)
    return hits / min(k, len(retrieved_at_k))


def recall_at_k(relevant: Set[str], retrieved: Sequence[str], k: int) -> float:
    """Return recall@k for a single query."""

    if not relevant:
        return 0.0
    retrieved_at_k = retrieved[:k]
    hits = sum(1 for doc_id in retrieved_at_k if doc_id in relevant)
    return hits / len(relevant)


def coverage_score(relevant: Set[str], retrieved: Sequence[str]) -> float:
    """Return the fraction of relevant documents retrieved at any rank."""

    if not relevant:
        return 0.0
    hits = sum(1 for doc_id in retrieved if doc_id in relevant)
    return hits / len(relevant)


def evaluate_retrieval(
    relevance_judgements: Mapping[str, Set[str]],
    retrieval_results: Mapping[str, Sequence[str]],
    k_values: Sequence[int] = (1, 3, 5),
) -> Dict[str, Dict[int, float]]:
    """Return aggregate retrieval metrics for multiple queries.

    Parameters
    ----------
    relevance_judgements:
        Mapping from query identifiers to the set of relevant document IDs.
    retrieval_results:
        Mapping from query identifiers to ranked document identifiers returned by
        the retriever.
    k_values:
        Cut-off points for which precision and recall should be computed.
    """

    metrics: Dict[str, Dict[int, float]] = {"precision": {}, "recall": {}}
    coverage_scores: List[float] = []
    for k in k_values:
        precision_values: List[float] = []
        recall_values: List[float] = []
        for query_id, relevant_docs in relevance_judgements.items():
            retrieved_docs = retrieval_results.get(query_id, [])
            precision_values.append(precision_at_k(relevant_docs, retrieved_docs, k))
            recall_values.append(recall_at_k(relevant_docs, retrieved_docs, k))
            coverage_scores.append(coverage_score(relevant_docs, retrieved_docs))
        metrics["precision"][k] = (
            sum(precision_values) / len(precision_values) if precision_values else 0.0
        )
        metrics["recall"][k] = (
            sum(recall_values) / len(recall_values) if recall_values else 0.0
        )
    metrics["coverage"] = {
        0: sum(coverage_scores) / len(coverage_scores) if coverage_scores else 0.0
    }
    return metrics


__all__ = ["precision_at_k", "recall_at_k", "coverage_score", "evaluate_retrieval"]
