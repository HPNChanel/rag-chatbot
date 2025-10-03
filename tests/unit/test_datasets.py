from pathlib import Path

import pytest

from evaluation.datasets import DatasetValidationError, load_dataset


def test_load_dataset_from_name():
    dataset = load_dataset("tiny")
    assert len(dataset.corpus) >= 5
    assert dataset.queries


def test_load_dataset_from_path(tmp_path: Path):
    base = tmp_path / "bench"
    base.mkdir()
    (base / "corpus.jsonl").write_text('{"id":"d1","text":"text","title":"t"}\n', encoding="utf-8")
    (base / "queries.jsonl").write_text('{"id":"q1","question":"?","gold_passages":["d1"],"gold_answers":["a"]}\n', encoding="utf-8")
    dataset = load_dataset(base)
    assert dataset.corpus[0].id == "d1"


def test_invalid_dataset(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_dataset(tmp_path)

    base = tmp_path / "bad"
    base.mkdir()
    (base / "corpus.jsonl").write_text("{}\n", encoding="utf-8")
    (base / "queries.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(DatasetValidationError):
        load_dataset(base)
