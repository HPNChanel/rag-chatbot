"""Utility script to compute retrieval and generation metrics on sample data."""
from __future__ import annotations

from pprint import pprint

from evaluation.generation_eval import corpus_bleu_score, meteor_score, rouge_l_score
from evaluation.retrieval_eval import evaluate_retrieval


SAMPLE_RELEVANCE = {
    "python": {"doc_python_basics", "doc_python_history"},
    "rust": {"doc_rust_memory"},
}

SAMPLE_RETRIEVAL = {
    "python": ["doc_python_basics", "doc_python_history", "doc_random"],
    "rust": ["doc_rust_memory", "doc_python_basics"],
}

SAMPLE_REFERENCES = {
    "python": "Python is popular for its readable syntax and large ecosystem.",
    "rust": "Rust emphasises memory safety without sacrificing performance.",
}

SAMPLE_GENERATIONS = {
    "python": "Python is popular thanks to its readable syntax and packages. [doc_python_basics]",
    "rust": "Rust emphasises memory safety and speed. [doc_rust_memory]",
}


def run_benchmark() -> None:
    """Execute retrieval and generation evaluation on the bundled samples."""

    retrieval_metrics = evaluate_retrieval(SAMPLE_RELEVANCE, SAMPLE_RETRIEVAL, (1, 2, 3))

    references = list(SAMPLE_REFERENCES.values())
    generations = list(SAMPLE_GENERATIONS.values())
    bleu = corpus_bleu_score(references, generations)
    rouge_l = rouge_l_score(references, generations)
    meteor = meteor_score(references, generations)

    print("Retrieval metrics:")
    pprint(retrieval_metrics)
    print("\nGeneration metrics:")
    pprint({
        "bleu": bleu,
        "rouge_l": rouge_l,
        "meteor": meteor,
    })


if __name__ == "__main__":  # pragma: no cover - script entry point
    run_benchmark()
