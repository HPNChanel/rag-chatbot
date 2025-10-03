from data_ingestion.loader import Document
from generation.citation_pipeline import CitationFirstPipeline, GeneratedAnswer


class MockGenerator:
    """Deterministic generator used for testing pipelines."""

    def __init__(self) -> None:
        self.last_prompt: str | None = None
        self.last_context: list[str] | None = None

    def generate(
        self,
        prompt: str,
        context: list[str] | None = None,
        max_tokens: int | None = None,
    ) -> str:
        self.last_prompt = prompt
        self.last_context = list(context or [])
        return "Python is versatile. [doc1] [doc2]"


def test_citation_pipeline_formats_context() -> None:
    generator = MockGenerator()
    pipeline = CitationFirstPipeline(generator)
    documents = [
        Document(doc_id="doc1", content="Python enables rapid development", metadata={}),
        Document(doc_id="doc2", content="It has an extensive ecosystem", metadata={}),
    ]

    result = pipeline.generate("Why is Python popular?", documents)

    assert isinstance(result, GeneratedAnswer)
    assert result.citations == ["doc1", "doc2"]
    assert generator.last_context == [
        "[doc1] Python enables rapid development",
        "[doc2] It has an extensive ecosystem",
    ]
    assert "Question" in generator.last_prompt
