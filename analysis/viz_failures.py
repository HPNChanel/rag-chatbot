"""Visualisation utilities for failure analysis."""

from __future__ import annotations

import html
import re
from pathlib import Path
from typing import Iterable, List, Mapping, Sequence


def highlight_hallucinations(answer: str, support_texts: Sequence[str]) -> str:
    """Highlight potential hallucinated spans using a unigram heuristic."""

    if not answer:
        return answer
    support_tokens = {token for text in support_texts for token in _tokenise(text)}
    tokens = answer.split()
    highlighted: List[str] = []
    for token in tokens:
        clean = _normalise(token)
        if clean and clean not in support_tokens and len(clean) > 3:
            highlighted.append(
                f'<mark class="hallucination">{html.escape(token)}</mark>'
            )
        else:
            highlighted.append(html.escape(token))
    return " ".join(highlighted)


def build_markdown_table(failures: Sequence[Mapping[str, object]]) -> str:
    """Return a Markdown representation for the provided failure cases."""

    if not failures:
        return "No failures detected."
    lines = [
        "| Query | Gold Answer(s) | Generated Answer | Retrieved Docs | Categories |",
        "| --- | --- | --- | --- | --- |",
    ]
    for failure in failures:
        question = _escape_markdown(str(failure.get("question", "")))
        gold_answers = "<br/>".join(
            _escape_markdown(answer) for answer in failure.get("gold_answers", [])
        )
        docs = "<br/>".join(
            _format_doc(doc) for doc in failure.get("retrieved_docs", [])
        )
        categories = ", ".join(failure.get("error_categories", []))
        answer = highlight_hallucinations(
            str(failure.get("generated_answer", "")),
            _collect_support_texts(failure),
        )
        lines.append(
            f"| {question} | {gold_answers} | {answer} | {docs} | {categories} |"
        )
    return "\n".join(lines)


def build_html_table(failures: Sequence[Mapping[str, object]]) -> str:
    """Return a standalone HTML document visualising the failures."""

    if not failures:
        return "<html><body><p>No failures detected.</p></body></html>"
    rows: List[str] = []
    for failure in failures:
        question = html.escape(str(failure.get("question", "")))
        gold_answers = "<br/>".join(
            html.escape(answer) for answer in failure.get("gold_answers", [])
        )
        docs = "<br/>".join(
            _format_doc_html(doc) for doc in failure.get("retrieved_docs", [])
        )
        categories = ", ".join(
            html.escape(cat) for cat in failure.get("error_categories", [])
        )
        answer = highlight_hallucinations(
            str(failure.get("generated_answer", "")),
            _collect_support_texts(failure),
        )
        rows.append(
            "<tr>"
            f"<td>{question}</td>"
            f"<td>{gold_answers}</td>"
            f"<td>{answer}</td>"
            f"<td>{docs}</td>"
            f"<td>{categories}</td>"
            "</tr>"
        )
    table = """
    <html>
    <head>
      <style>
        body {{ font-family: Arial, sans-serif; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; vertical-align: top; }}
        th {{ background-color: #f4f4f4; }}
        mark.hallucination {{ background-color: #ffe0e0; }}
        span.doc-miss {{ color: #b00020; font-weight: 600; }}
        span.doc-hit {{ color: #1b5e20; font-weight: 600; }}
      </style>
    </head>
    <body>
      <h2>Failure Cases</h2>
      <table>
        <thead>
          <tr>
            <th>Query</th>
            <th>Gold Answer(s)</th>
            <th>Generated Answer</th>
            <th>Retrieved Docs</th>
            <th>Categories</th>
          </tr>
        </thead>
        <tbody>
          {rows}
        </tbody>
      </table>
    </body>
    </html>
    """.strip()
    return table.format(rows="\n".join(rows))


def export_failure_visualisations(
    failures: Sequence[Mapping[str, object]],
    run_dir: Path,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    markdown = build_markdown_table(failures)
    html_doc = build_html_table(failures)
    (run_dir / "failures.md").write_text(markdown, encoding="utf-8")
    (run_dir / "failures.html").write_text(html_doc, encoding="utf-8")


def _tokenise(text: str) -> List[str]:
    return [token.lower() for token in re.findall(r"[a-z0-9]+", text)]


def _normalise(token: str) -> str:
    return "".join(ch for ch in token.lower() if ch.isalnum())


def _collect_support_texts(failure: Mapping[str, object]) -> List[str]:
    support: List[str] = []
    support.extend(str(answer) for answer in failure.get("gold_answers", []))
    for doc in failure.get("retrieved_docs", []):
        support.append(str(doc.get("content", "")))
    return support


def _format_doc(doc: Mapping[str, object]) -> str:
    prefix = "✅" if doc.get("is_gold") else "⚠️"
    if doc.get("selected"):
        prefix += "★"
    content = _escape_markdown(str(doc.get("content", "")))
    return f"{prefix} {doc.get('doc_id')} — {content[:160]}"


def _format_doc_html(doc: Mapping[str, object]) -> str:
    status_class = "doc-hit" if doc.get("is_gold") else "doc-miss"
    prefix = "✅" if doc.get("is_gold") else "⚠️"
    if doc.get("selected"):
        prefix += "★"
    return (
        f"<span class=\"{status_class}\">{prefix} {html.escape(str(doc.get('doc_id')))}</span>"
        f"<br/><small>{html.escape(str(doc.get('content', '')))}</small>"
    )


def _escape_markdown(text: str) -> str:
    return text.replace("|", "\\|")


__all__ = [
    "highlight_hallucinations",
    "build_markdown_table",
    "build_html_table",
    "export_failure_visualisations",
]
