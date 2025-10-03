from evaluation.metrics import (
    aggregate_generation_metrics,
    aggregate_retrieval_metrics,
    mean_and_std,
    precision_at_k,
    recall_at_k,
)


def test_precision_and_recall_at_k():
    relevant = ["a", "b"]
    retrieved = ["a", "c", "b"]
    assert precision_at_k(relevant, retrieved, 2) == 0.5
    assert recall_at_k(relevant, retrieved, 2) == 0.5


def test_aggregate_retrieval_metrics():
    judgements = {"q1": ["d1"], "q2": ["d2"]}
    results = {"q1": ["d1", "d3"], "q2": ["d3", "d2"]}
    metrics = aggregate_retrieval_metrics(judgements, results, k_values=[1, 2])
    assert metrics["precision"][1] == 0.5
    assert metrics["recall"][2] == 1.0


def test_generation_metrics():
    references = {"q1": ["mars is red"]}
    predictions = {"q1": "mars is red"}
    metrics = aggregate_generation_metrics(references, predictions)
    assert metrics["rougeL"] > 0.9


def test_mean_and_std():
    mean, std = mean_and_std([1.0, 3.0])
    assert mean == 2.0
    assert round(std, 6) == round(((1**2 + 1**2) / 2) ** 0.5, 6)
