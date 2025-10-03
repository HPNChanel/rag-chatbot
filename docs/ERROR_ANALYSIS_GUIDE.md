# Error Analysis Guide

This guide explains how to interpret the automated failure analysis produced by the RAGX benchmarking stack and how to extend it for custom projects. The companion tools live under the `analysis/` package and integrate with the `ragx-analyze` CLI command.

## Failure Categories

The error analyzer assigns one or more categories to each failed query using lightweight heuristics and semantic similarity signals:

| Category | Description | Typical Cause |
| --- | --- | --- |
| Retrieval Miss | Gold evidence was never retrieved, so generation had no way to answer correctly. | Index gaps, tokenisation mismatch, restrictive filters. |
| Citation Error | The generated answer cites the wrong document ID or omits citations when relevant evidence was retrieved. | Prompt formatting bugs, citation post-processing issues. |
| Hallucination | The answer contains claims with little lexical overlap with the supporting context. | Aggressive decoding parameters, prompt drift, insufficient grounding. |
| Coverage Gap | Retrieved documents provide redundant information leading to poor coverage of the question. | Overly similar top-k results, missing diversification, re-ranker misconfiguration. |
| Paraphrase Miss | Relevant gold documents were retrieved but lexical metrics fail because the language is paraphrased. | Evaluation relying on string overlap, paraphrased corpora, multilingual content. |

A single query may trigger multiple categories. The analyzer surfaces a `hallucination_score`, redundancy estimates, and citation diagnostics inside `failures.jsonl` for deeper inspection.

## Reviewing Failure Reports

1. Run `ragx-analyze --run-dir runs/<run_id> --redact` to build the artefacts.
2. Inspect `runs/<run_id>/failures.md` (Markdown) or `failures.html` (rich table) for redacted examples. Hallucinated spans are highlighted with a pink `<mark>` tag.
3. Open `report.md` and navigate to the **Error Analysis** section. A bar chart summarises category counts together with sample failures. The section links back to the full log for team reviews.

## Extending the Rules

* Add new category heuristics by updating `analysis/errors.py`. Each rule can append to the `categories` list returned by `categorize_failure`.
* Use embedding similarity or domain-specific checks by overriding the `similarity_threshold` when instantiating `ErrorAnalyzer`.
* Integrate additional metadata (e.g., retrieval scores, user feedback) by enriching the payload before writing to `failures.jsonl`.

### Custom Pipelines

For pipelines with bespoke retrieval stacks or corpora:

1. Populate `doc_lookup` with document text when creating `ErrorAnalyzer` so that coverage and hallucination heuristics have sufficient evidence.
2. Provide per-query gold labels in the format consumed by `extract_failure_cases` (`gold_answers`, `gold_doc_ids`).
3. Optionally attach domain tags to each failure to power downstream dashboards.

## Safety and Privacy Redaction

Redaction is controlled via the `safety` block in `config.used.yaml` or the `--redact` CLI flag:

```yaml
safety:
  enable_redaction: true
  pii: true          # redact emails, phone numbers, SSNs, credit cards, basic NER
  sensitive_terms:
    - Project Phoenix
    - customer_id
```

The analyzer applies `redact_pii` and `redact_sensitive_terms` to queries, answers, and retrieved documents before writing any artefacts. This keeps logs safe to share across teams or vendors. Extend the list of `sensitive_terms` to mask organisation-specific code names or confidential entities.

## Sharing Insights Safely

* Always run analyses with redaction enabled for external reviews.
* Store generated reports in access-controlled locations; the Markdown files reference only redacted text.
* When adding new failure cases manually, run them through `TextRedactor` to maintain consistency with automated reports.

By following these guidelines the team can rapidly triage RAG regressions while meeting privacy, safety, and governance requirements.
