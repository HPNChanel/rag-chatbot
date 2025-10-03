"""Dataset utilities for benchmark experiments."""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

from pydantic import BaseModel, Field, ValidationError


class CorpusRecord(BaseModel):
    id: str
    text: str
    title: str | None = None
    source: str | None = None


class QueryRecord(BaseModel):
    id: str
    question: str
    gold_passages: List[str] = Field(default_factory=list)
    gold_answers: List[str] = Field(default_factory=list)


@dataclass
class Dataset:
    corpus: List[CorpusRecord]
    queries: List[QueryRecord]


class DatasetValidationError(RuntimeError):
    """Raised when dataset files do not conform to the expected schema."""


def _read_jsonl(path: Path) -> List[dict]:
    records: List[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:  # pragma: no cover - guard
                raise DatasetValidationError(
                    f"Failed to parse JSON on line {line_number} of {path}: {exc}"
                ) from exc
    return records


def _validate_records(records: Iterable[dict], model: type[BaseModel]) -> List[BaseModel]:
    validated: List[BaseModel] = []
    for record in records:
        try:
            validated.append(model.model_validate(record))
        except ValidationError as exc:
            raise DatasetValidationError(str(exc)) from exc
    return validated


def load_dataset(dataset_path: str | Path, *, sample: Optional[int] = None) -> Dataset:
    """Load queries/corpus from ``dataset_path``.

    ``dataset_path`` may be a directory name under ``data/benchmarks`` or an
    absolute/relative path. When ``sample`` is provided the queries are randomly
    sampled without replacement using ``random.sample``.
    """

    base_path = _resolve_dataset_path(dataset_path)
    queries_path = base_path / "queries.jsonl"
    corpus_path = base_path / "corpus.jsonl"
    if not queries_path.exists() or not corpus_path.exists():
        raise FileNotFoundError(
            "Dataset directory must contain queries.jsonl and corpus.jsonl"
        )
    query_payloads = _read_jsonl(queries_path)
    corpus_payloads = _read_jsonl(corpus_path)
    queries = _validate_records(query_payloads, QueryRecord)
    corpus = _validate_records(corpus_payloads, CorpusRecord)
    queries_list = list(queries)
    if sample is not None and 0 < sample < len(queries_list):
        queries_list = random.sample(queries_list, sample)
    return Dataset(corpus=list(corpus), queries=queries_list)


def _resolve_dataset_path(dataset_path: str | Path) -> Path:
    path = Path(dataset_path)
    if path.exists():
        return path
    project_path = Path(__file__).resolve().parent.parent
    candidate = project_path / "data" / "benchmarks" / str(dataset_path)
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Could not find dataset at {dataset_path}")


__all__ = [
    "CorpusRecord",
    "QueryRecord",
    "Dataset",
    "DatasetValidationError",
    "load_dataset",
]
