"""Failure categorisation utilities for RAG runs."""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .redaction import TextRedactor


CATEGORIES = [
    "Retrieval Miss",
    "Citation Error",
    "Hallucination",
    "Coverage Gap",
    "Paraphrase Miss",
]


@dataclass
class RetrievedDoc:
    """Lightweight container for retrieved documents."""

    doc_id: str
    content: str
    score: Optional[float] = None
    selected: bool = False
    is_gold: bool = False


class ErrorAnalyzer:
    """Analyse raw experiment results to identify systematic failure modes."""

    def __init__(
        self,
        run_dir: Path | str,
        *,
        redactor: Optional[TextRedactor] = None,
        doc_lookup: Optional[Mapping[str, str]] = None,
        similarity_threshold: float = 0.6,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.failures_path = self.run_dir / "failures.jsonl"
        self.redactor = redactor
        self.doc_lookup = dict(doc_lookup or {})
        self.similarity_threshold = similarity_threshold

    # ------------------------------------------------------------------
    # Public API
    def categorize_failure(
        self,
        query: str,
        gold_answers: Mapping[str, Sequence[str]] | Sequence[str] | None,
        retrieved_docs: Sequence[Mapping[str, object]] | Sequence[RetrievedDoc],
        generated_answer: str,
    ) -> Dict[str, object]:
        """Assign heuristic failure categories for a single query.

        Parameters
        ----------
        query:
            The user query.
        gold_answers:
            Either a sequence of acceptable answer strings or a mapping containing
            ``"answers"``/``"gold_answers"`` and optionally ``"gold_doc_ids"``.
        retrieved_docs:
            Sequence of retrieved document payloads (dicts or :class:`RetrievedDoc`).
        generated_answer:
            The model generated answer (potentially containing citations).
        """

        answers, gold_doc_ids = self._normalise_gold_answers(gold_answers)
        docs = [self._coerce_retrieved(doc, gold_doc_ids) for doc in retrieved_docs]
        categories: List[str] = []
        signals: Dict[str, object] = {}

        retrieved_ids = {doc.doc_id for doc in docs}
        if gold_doc_ids and not retrieved_ids.intersection(gold_doc_ids):
            categories.append("Retrieval Miss")

        citation_ids = self._extract_citations(generated_answer, docs)
        if citation_ids:
            invalid = [cid for cid in citation_ids if cid not in retrieved_ids]
            if invalid:
                categories.append("Citation Error")
                signals["invalid_citations"] = invalid
        else:
            # When a gold document was retrieved but not cited we flag it as an error.
            if any(doc.is_gold for doc in docs):
                categories.append("Citation Error")
                signals["missing_citations"] = True

        hallucination_score = self._hallucination_score(generated_answer, answers, docs)
        signals["hallucination_score"] = hallucination_score
        if hallucination_score < 0.5:
            categories.append("Hallucination")

        if len(docs) >= 2:
            redundancy = self._average_similarity([doc.content for doc in docs])
            unique_contents = {doc.content.lower() for doc in docs if doc.content}
            coverage_gap = bool(
                unique_contents and len(unique_contents) <= len(docs) // 2
            )
            signals["doc_redundancy"] = redundancy
            if redundancy > 0.85 or coverage_gap:
                categories.append("Coverage Gap")

        paraphrase_hit = self._paraphrase_detected(answers, docs)
        if paraphrase_hit:
            categories.append("Paraphrase Miss")
            signals["paraphrase_support"] = True

        return {"categories": list(dict.fromkeys(categories)), "signals": signals}

    def extract_failure_cases(
        self,
        results_raw: Path | str | Iterable[Mapping[str, object]],
        gold_labels: Mapping[str, MutableMapping[str, object]],
    ) -> List[Dict[str, object]]:
        """Extract and persist failed queries with category annotations."""

        records = self._load_results(results_raw)
        failures: List[Dict[str, object]] = []
        for payload in records:
            query_id = str(payload.get("query_id"))
            gold_info = gold_labels.get(query_id, {})
            answers = gold_info.get("gold_answers") or gold_info.get("answers") or []
            generated_answer = str(payload.get("generated_answer", ""))
            if self._is_success(generated_answer, answers):
                continue
            retrieved_docs = self._build_retrieved_docs(payload, gold_info)
            categorisation = self.categorize_failure(
                str(payload.get("question", "")),
                gold_info,
                retrieved_docs,
                generated_answer,
            )
            if not categorisation["categories"]:
                continue
            failure_record = {
                "query_id": query_id,
                "question": self._redact_text(str(payload.get("question", ""))),
                "gold_answers": [self._redact_text(ans) for ans in answers],
                "generated_answer": self._redact_text(generated_answer),
                "error_categories": categorisation["categories"],
                "signals": categorisation["signals"],
                "retrieved_docs": [
                    {
                        "doc_id": doc.doc_id,
                        "score": doc.score,
                        "selected": doc.selected,
                        "is_gold": doc.is_gold,
                        "content": self._redact_text(doc.content),
                    }
                    for doc in retrieved_docs
                ],
            }
            failures.append(failure_record)
        self._write_failures(failures)
        return failures

    # ------------------------------------------------------------------
    # Internal helpers
    def _load_results(
        self, results_raw: Path | str | Iterable[Mapping[str, object]]
    ) -> List[MutableMapping[str, object]]:
        if isinstance(results_raw, (str, Path)):
            path = Path(results_raw)
            if not path.exists():
                raise FileNotFoundError(f"Results file {path} does not exist")
            payloads: List[MutableMapping[str, object]] = []
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    payloads.append(json.loads(line))
            return payloads
        return [dict(item) for item in results_raw]

    def _coerce_retrieved(
        self,
        payload: Mapping[str, object] | RetrievedDoc,
        gold_doc_ids: Sequence[str] | set[str],
    ) -> RetrievedDoc:
        if isinstance(payload, RetrievedDoc):
            return payload
        doc_id = (
            str(payload.get("doc_id")) if isinstance(payload, Mapping) else str(payload)
        )
        score = None
        selected = False
        if isinstance(payload, Mapping):
            raw_score = payload.get("score")
            if isinstance(raw_score, (int, float)):
                score = float(raw_score)
            selected = bool(payload.get("selected", False))
            if not selected:
                citations = payload.get("citations")
                if isinstance(citations, Sequence) and doc_id in citations:
                    selected = True
        content = self.doc_lookup.get(
            doc_id,
            str(payload.get("content", "")) if isinstance(payload, Mapping) else "",
        )
        return RetrievedDoc(
            doc_id=doc_id,
            content=content,
            score=score,
            selected=selected,
            is_gold=doc_id in set(gold_doc_ids),
        )

    def _build_retrieved_docs(
        self, payload: Mapping[str, object], gold_info: Mapping[str, object]
    ) -> List[RetrievedDoc]:
        citations = set(
            payload.get("citations") or payload.get("selected_context") or []
        )
        gold_doc_ids = set(
            gold_info.get("gold_doc_ids")
            or gold_info.get("gold_passages")
            or gold_info.get("passages")
            or []
        )
        docs: List[RetrievedDoc] = []
        for item in payload.get("retrieved", []) or []:
            if isinstance(item, dict):
                doc_id = str(item.get("doc_id"))
                score = item.get("score")
            else:
                doc_id = str(item)
                score = None
            content = self.doc_lookup.get(doc_id, "")
            docs.append(
                RetrievedDoc(
                    doc_id=doc_id,
                    content=content,
                    score=float(score) if isinstance(score, (int, float)) else None,
                    selected=doc_id in citations,
                    is_gold=doc_id in gold_doc_ids,
                )
            )
        if not docs and payload.get("selected_context"):
            for doc_id in payload.get("selected_context", []):
                doc_id = str(doc_id)
                docs.append(
                    RetrievedDoc(
                        doc_id=doc_id,
                        content=self.doc_lookup.get(doc_id, ""),
                        score=None,
                        selected=True,
                        is_gold=doc_id in gold_doc_ids,
                    )
                )
        return docs

    def _normalise_gold_answers(
        self, gold_answers: Mapping[str, Sequence[str]] | Sequence[str] | None
    ) -> tuple[List[str], set[str]]:
        answers: List[str] = []
        passages: set[str] = set()
        if gold_answers is None:
            return answers, passages
        if isinstance(gold_answers, Mapping):
            answers = list(
                gold_answers.get("gold_answers") or gold_answers.get("answers") or []
            )
            passages = set(
                gold_answers.get("gold_doc_ids")
                or gold_answers.get("gold_passages")
                or gold_answers.get("passages")
                or []
            )
        else:
            answers = list(gold_answers)
        return answers, passages

    def _extract_citations(
        self, generated_answer: str, docs: Sequence[RetrievedDoc]
    ) -> List[str]:
        pattern = re.compile(r"\[(?P<cid>[^\[\]]+)\]")
        citations = pattern.findall(generated_answer or "")
        if citations:
            return citations
        # fall back to selected flag when explicit citations missing
        selected = [doc.doc_id for doc in docs if doc.selected]
        return selected

    def _hallucination_score(
        self,
        generated_answer: str,
        gold_answers: Sequence[str],
        docs: Sequence[RetrievedDoc],
    ) -> float:
        generated_tokens = self._tokenise(generated_answer)
        if not generated_tokens:
            return 0.0
        support_tokens: Counter[str] = Counter()
        for answer in gold_answers:
            support_tokens.update(self._tokenise(answer))
        for doc in docs:
            support_tokens.update(self._tokenise(doc.content))
        supported = sum(1 for token in generated_tokens if token in support_tokens)
        return supported / max(len(generated_tokens), 1)

    def _average_similarity(self, texts: Sequence[str]) -> float:
        clean_texts = [text for text in texts if text]
        if len(clean_texts) < 2:
            return 0.0
        vectorizer = TfidfVectorizer(stop_words="english")
        matrix = vectorizer.fit_transform(clean_texts)
        similarities = cosine_similarity(matrix)
        total = 0.0
        count = 0
        for i in range(len(clean_texts)):
            for j in range(i + 1, len(clean_texts)):
                total += float(similarities[i, j])
                count += 1
        return total / max(count, 1)

    def _paraphrase_detected(
        self, gold_answers: Sequence[str], docs: Sequence[RetrievedDoc]
    ) -> bool:
        if not gold_answers or not docs:
            return False
        combined_answers = " ".join(gold_answers)
        for doc in docs:
            if not doc.content:
                continue
            lexical_gold = self._lexical_overlap(doc.content, combined_answers)
            if doc.is_gold and lexical_gold < 0.3:
                return True
            for answer in gold_answers:
                semantic = self._semantic_similarity(doc.content, answer)
                lexical = self._lexical_overlap(doc.content, answer)
                if semantic >= self.similarity_threshold and lexical < 0.5:
                    return True
        return False

    def _semantic_similarity(self, text_a: str, text_b: str) -> float:
        if not text_a or not text_b:
            return 0.0
        vectorizer = TfidfVectorizer(stop_words="english")
        matrix = vectorizer.fit_transform([text_a, text_b])
        similarity = cosine_similarity(matrix[0:1], matrix[1:2])
        return float(similarity[0, 0])

    def _lexical_overlap(self, text_a: str, text_b: str) -> float:
        tokens_a = set(self._tokenise(text_a))
        tokens_b = set(self._tokenise(text_b))
        if not tokens_a or not tokens_b:
            return 0.0
        return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)

    def _tokenise(self, text: str) -> List[str]:
        return re.findall(r"[a-z0-9]+", text.lower())

    def _is_success(self, generated_answer: str, gold_answers: Sequence[str]) -> bool:
        if not gold_answers:
            return False
        generated_lower = generated_answer.lower()
        for answer in gold_answers:
            candidate = answer.lower()
            if candidate in generated_lower:
                return True
            overlap = self._lexical_overlap(candidate, generated_lower)
            if overlap >= 0.7:
                return True
        return False

    def _redact_text(self, text: str) -> str:
        if not text:
            return text
        if self.redactor is None:
            return text
        return self.redactor.redact(text)

    def _write_failures(self, failures: Sequence[Mapping[str, object]]) -> None:
        if not failures:
            # ensure previous runs do not persist stale files
            if self.failures_path.exists():
                self.failures_path.unlink()
            return
        with self.failures_path.open("w", encoding="utf-8") as handle:
            for failure in failures:
                handle.write(json.dumps(failure) + "\n")


__all__ = ["ErrorAnalyzer", "RetrievedDoc", "CATEGORIES"]
