from __future__ import annotations

from analysis.errors import ErrorAnalyzer


def _analyzer(tmp_path, doc_lookup=None):
    return ErrorAnalyzer(tmp_path, doc_lookup=doc_lookup or {})


def test_retrieval_miss_detected(tmp_path):
    analyzer = _analyzer(tmp_path)
    result = analyzer.categorize_failure(
        query="Who discovered penicillin?",
        gold_answers={
            "gold_answers": ["Alexander Fleming"],
            "gold_doc_ids": ["doc_gold"],
        },
        retrieved_docs=[
            {"doc_id": "doc1", "content": "Louis Pasteur pioneered vaccines."}
        ],
        generated_answer="It was Pasteur according to [doc1].",
    )
    assert "Retrieval Miss" in result["categories"]


def test_citation_error_detected(tmp_path):
    analyzer = _analyzer(tmp_path)
    result = analyzer.categorize_failure(
        query="Who wrote Hamlet?",
        gold_answers={
            "gold_answers": ["William Shakespeare"],
            "gold_doc_ids": ["doc_hamlet"],
        },
        retrieved_docs=[
            {"doc_id": "doc_hamlet", "content": "Hamlet is a Shakespeare play."}
        ],
        generated_answer="Hamlet was by Shakespeare [doc_unknown]",
    )
    assert "Citation Error" in result["categories"]


def test_hallucination_detected(tmp_path):
    analyzer = _analyzer(tmp_path)
    result = analyzer.categorize_failure(
        query="What is the capital of Australia?",
        gold_answers={"gold_answers": ["Canberra"], "gold_doc_ids": ["doc_canberra"]},
        retrieved_docs=[
            {
                "doc_id": "doc_canberra",
                "content": "Canberra is the capital city of Australia.",
            }
        ],
        generated_answer="The capital is Sydney with 10 million residents.",
    )
    assert "Hallucination" in result["categories"]


def test_coverage_gap_detected(tmp_path):
    doc_lookup = {
        "doc_a": "Paris is the capital of France.",
        "doc_b": "Paris is the capital of France.",
    }
    analyzer = _analyzer(tmp_path, doc_lookup=doc_lookup)
    result = analyzer.categorize_failure(
        query="What is the capital of France?",
        gold_answers={"gold_answers": ["Paris"], "gold_doc_ids": ["doc_a"]},
        retrieved_docs=[
            {"doc_id": "doc_a", "content": doc_lookup["doc_a"]},
            {"doc_id": "doc_b", "content": doc_lookup["doc_b"]},
        ],
        generated_answer="It is Paris.",
    )
    assert "Coverage Gap" in result["categories"]


def test_paraphrase_miss_detected(tmp_path):
    doc_lookup = {
        "doc_gold": "France's governmental seat lies in Paris beside the Seine river and houses national ministries.",
    }
    analyzer = _analyzer(tmp_path, doc_lookup=doc_lookup)
    result = analyzer.categorize_failure(
        query="Where is the government of France located?",
        gold_answers={
            "gold_answers": ["Paris is the capital of France."],
            "gold_doc_ids": ["doc_gold"],
        },
        retrieved_docs=[
            {"doc_id": "doc_gold", "content": doc_lookup["doc_gold"], "is_gold": True}
        ],
        generated_answer="The institutions meet in Lyon.",
    )
    assert "Paraphrase Miss" in result["categories"]
