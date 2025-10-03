"""Evaluation utilities for text generation quality."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from statistics import mean
from typing import Dict, List, Sequence

try:  # pragma: no cover - optional dependency
    from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
except Exception:  # pragma: no cover
    SmoothingFunction = None  # type: ignore
    corpus_bleu = None  # type: ignore

def corpus_bleu_score(references: Sequence[str], hypotheses: Sequence[str]) -> float:
    """Return the corpus-level BLEU score."""

    if not references or not hypotheses:
        return 0.0
    reference_list = [[ref.split()] for ref in references]
    hypothesis_tokens = [hyp.split() for hyp in hypotheses]
    if corpus_bleu is None or SmoothingFunction is None:  # pragma: no cover - fallback
        return _simple_bleu(reference_list, hypothesis_tokens)
    smoothing = SmoothingFunction().method1
    try:
        return corpus_bleu(reference_list, hypothesis_tokens, smoothing_function=smoothing)
    except TypeError:  # pragma: no cover - fallback for Python 3.12 Fraction API change
        return _simple_bleu(reference_list, hypothesis_tokens)


def rouge_l_score(references: Sequence[str], hypotheses: Sequence[str]) -> float:
    """Return the average ROUGE-L F-measure using a lightweight implementation."""

    if not references or not hypotheses:
        return 0.0
    scores = []
    for ref, hyp in zip(references, hypotheses):
        ref_tokens = ref.split()
        hyp_tokens = hyp.split()
        if not ref_tokens or not hyp_tokens:
            scores.append(0.0)
            continue
        lcs = _lcs_length(ref_tokens, hyp_tokens)
        recall = lcs / len(ref_tokens)
        precision = lcs / len(hyp_tokens)
        if precision == 0 or recall == 0:
            scores.append(0.0)
            continue
        beta_sq = (recall / precision) ** 2
        scores.append(((1 + beta_sq) * precision * recall) / (recall + beta_sq * precision))
    return mean(scores) if scores else 0.0


def _simple_bleu(
    references: Sequence[Sequence[Sequence[str]]], hypothesis_tokens: Sequence[Sequence[str]]
) -> float:
    """Fallback BLEU implementation used when NLTK is unavailable."""

    scores: List[float] = []
    for refs, hyp in zip(references, hypothesis_tokens):
        ref = refs[0] if refs else []
        if not ref or not hyp:
            scores.append(0.0)
            continue
        ref_counts: Dict[str, int] = {}
        for token in ref:
            ref_counts[token] = ref_counts.get(token, 0) + 1
        matches = 0
        used: Dict[str, int] = {}
        for token in hyp:
            limit = ref_counts.get(token, 0)
            consumed = used.get(token, 0)
            if consumed < limit:
                used[token] = consumed + 1
                matches += 1
        precision = matches / len(hyp)
        brevity = min(1.0, len(hyp) / len(ref))
        scores.append(precision * brevity)
    return mean(scores) if scores else 0.0


def _lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    """Compute the length of the longest common subsequence."""

    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[-1][-1]


def meteor_score(references: Sequence[str], hypotheses: Sequence[str], *, alpha: float = 0.85) -> float:
    """Compute a simplified METEOR score averaged over all pairs.

    The reference implementation depends on WordNet which is not available in the
    execution environment. Instead we approximate METEOR using the harmonic mean
    of unigram precision and recall with configurable ``alpha``.
    """

    if not references or not hypotheses:
        return 0.0

    scores: List[float] = []
    for ref, hyp in zip(references, hypotheses):
        ref_tokens = ref.split()
        hyp_tokens = hyp.split()
        if not ref_tokens or not hyp_tokens:
            scores.append(0.0)
            continue
        ref_counts: Dict[str, int] = {}
        for token in ref_tokens:
            ref_counts[token] = ref_counts.get(token, 0) + 1
        match_count = 0
        used: Dict[str, int] = {}
        for token in hyp_tokens:
            limit = ref_counts.get(token, 0)
            consumed = used.get(token, 0)
            if consumed < limit:
                used[token] = consumed + 1
                match_count += 1
        precision = match_count / len(hyp_tokens)
        recall = match_count / len(ref_tokens)
        if precision == 0 or recall == 0:
            scores.append(0.0)
            continue
        denom = (1 - alpha) * precision + alpha * recall
        scores.append((precision * recall) / denom)
    return mean(scores) if scores else 0.0


@dataclass(slots=True)
class HumanEvalTemplate:
    """Structure used to serialise human evaluation tasks."""

    query: str
    reference_answer: str
    generated_answer: str
    citation_checklist: List[str]

    def to_json(self) -> str:
        """Serialise the template as JSON for annotators."""

        return json.dumps(asdict(self), ensure_ascii=False, indent=2)


def build_human_eval_payload(
    query: str,
    reference_answer: str,
    generated_answer: str,
    citations: Sequence[str],
) -> HumanEvalTemplate:
    """Return a :class:`HumanEvalTemplate` for manual annotations."""

    checklist = [
        f"Verify that citation {citation} supports the associated claim."
        for citation in citations
    ] or [
        "No citations supplied. Confirm whether the answer still appears factual."
    ]
    return HumanEvalTemplate(
        query=query,
        reference_answer=reference_answer,
        generated_answer=generated_answer,
        citation_checklist=checklist,
    )


__all__ = [
    "corpus_bleu_score",
    "rouge_l_score",
    "meteor_score",
    "HumanEvalTemplate",
    "build_human_eval_payload",
]
