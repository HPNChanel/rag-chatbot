from evaluation.generation_eval import corpus_bleu_score, meteor_score, rouge_l_score
from evaluation.retrieval_eval import coverage_score, precision_at_k, recall_at_k


def test_retrieval_metrics() -> None:
    relevant = {"doc1", "doc2"}
    retrieved = ["doc1", "doc3", "doc2"]

    assert precision_at_k(relevant, retrieved, 1) == 1.0
    assert precision_at_k(relevant, retrieved, 2) == 0.5
    assert recall_at_k(relevant, retrieved, 3) == 1.0
    assert coverage_score(relevant, retrieved) == 1.0


def test_generation_metrics() -> None:
    references = [
        "Python is a versatile language used for scripting and web development.",
        "Rust provides memory safety guarantees.",
    ]
    hypotheses = [
        "Python is versatile and great for scripting.",
        "Rust provides strong memory safety guarantees.",
    ]

    bleu = corpus_bleu_score(references, hypotheses)
    rouge_l = rouge_l_score(references, hypotheses)
    meteor = meteor_score(references, hypotheses)

    assert 0 < bleu <= 1
    assert 0 < rouge_l <= 1
    assert 0 < meteor <= 1
