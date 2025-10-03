"""Experiment runner for the RAG benchmarking suite."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import defaultdict
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import yaml

from data_ingestion.loader import Document
from evaluation.datasets import Dataset, load_dataset
from evaluation.metrics import aggregate_generation_metrics, aggregate_retrieval_metrics, mean_and_std
from evaluation.plots import plot_bar_metrics, plot_latency, plot_scatter
from evaluation.tabulate import save_metric_table
from experiments.seed import fix_seeds
from experiments.tracking import Tracker
from generation.dummy import EchoGenerator
from indexing.embedder import HashingEmbedder
from indexing.vector_store import FaissVectorStore
from reranking.coverage import CoverageAwareReranker
from repro.env_capture import write_environment_snapshot
from retrieval.bm25 import BM25Retriever
from retrieval.dense import DenseRetriever
from retrieval.prf import DualStagePRFRetriever

Config = Dict[str, Any]


def _load_config(path: Path) -> Config:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _apply_override(config: Config, key: str, value: str) -> None:
    parts = key.split(".")
    target = config
    for part in parts[:-1]:
        if part not in target or not isinstance(target[part], dict):
            target[part] = {}
        target = target[part]
    try:
        parsed_value = json.loads(value)
    except json.JSONDecodeError:
        parsed_value = value
    target[parts[-1]] = parsed_value


def _resolve_config(base_config: Config, overrides: Sequence[str]) -> Config:
    config = deepcopy(base_config)
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Override must be in key=value format: {override}")
        key, value = override.split("=", 1)
        _apply_override(config, key, value)
    return config


def _prepare_documents(dataset: Dataset) -> List[Document]:
    documents: List[Document] = []
    for record in dataset.corpus:
        documents.append(
            Document(
                doc_id=str(record.id),
                content=record.text,
                metadata={
                    "title": record.title or "",
                    "source": record.source or "",
                    "document_id": str(record.id),
                },
            )
        )
    return documents


def _create_retriever(config: Mapping[str, Any], documents: Sequence[Document]):
    retrieval_cfg = config.get("retrieval", {})
    retriever_type = retrieval_cfg.get("type", "bm25")
    top_k = retrieval_cfg.get("top_k", 5)
    if retriever_type == "faiss":
        embedder = HashingEmbedder(dim=128)
        store = FaissVectorStore(embedder, use_faiss=False)
        store.build(list(documents))
        base = DenseRetriever(store)
    elif retriever_type == "hybrid":
        bm25 = BM25Retriever(documents)
        embedder = HashingEmbedder(dim=128)
        store = FaissVectorStore(embedder, use_faiss=False)
        store.build(list(documents))
        dense = DenseRetriever(store)

        class HybridRetriever:
            def search(self, query: str, k: int = 5):
                lexical = bm25.search(query, k=k)
                dense_results = dense.search(query, k=k)
                scores = defaultdict(float)
                docs = {}
                for doc, score in lexical:
                    scores[doc.doc_id] += float(score)
                    docs[doc.doc_id] = doc
                for doc, score in dense_results:
                    scores[doc.doc_id] += float(score)
                    docs[doc.doc_id] = doc
                ranked = sorted(((docs[doc_id], score) for doc_id, score in scores.items()), key=lambda item: item[1], reverse=True)
                return ranked[:k]

        base = HybridRetriever()
    else:
        base = BM25Retriever(documents)
    prf_cfg = retrieval_cfg.get("prf", {})
    if prf_cfg.get("enabled"):
        base = DualStagePRFRetriever(
            base,
            feedback_depth=int(prf_cfg.get("feedback_k", 3)),
            expansion_terms=int(prf_cfg.get("expansion_terms", 10)),
            re_rank_strategy=str(prf_cfg.get("strategy", "sum")),
        )
    return base, top_k


def _create_generator(config: Mapping[str, Any]):
    generation_cfg = config.get("generation", {})
    provider = generation_cfg.get("provider", "mock")
    if provider != "mock":
        raise RuntimeError("Only the mock provider is supported in the benchmarking harness")
    return EchoGenerator()


def _run_single(config: Config, run_dir: Path, seed: int) -> Dict[str, Any]:
    fix_seeds(seed)
    sample = None
    if config.get("random_query_sample") and config.get("num_queries"):
        sample = int(config["num_queries"])
    dataset = load_dataset(config["dataset_path"], sample=sample)
    if config.get("num_queries") and not config.get("random_query_sample"):
        dataset.queries = dataset.queries[: int(config["num_queries"])]
    documents = _prepare_documents(dataset)
    retriever, top_k = _create_retriever(config, documents)
    generator = _create_generator(config)
    rerank_cfg = config.get("reranking", {})
    reranker = None
    if rerank_cfg.get("enabled"):
        reranker = CoverageAwareReranker(
            similarity_weight=float(rerank_cfg.get("lambda_similarity", 0.7)),
            diversity_bias=float(rerank_cfg.get("lambda_coverage", 0.15)),
        )
    results_path = run_dir / "results_raw.jsonl"
    references = {}
    predictions = {}
    retrieval_results = {}
    latencies = []
    with results_path.open("w", encoding="utf-8") as handle:
        for query in dataset.queries:
            start = time.perf_counter()
            retrieved = retriever.search(query.question, k=top_k)
            retrieval_time = time.perf_counter()
            if reranker is not None:
                reranked = reranker.rerank(retrieved, top_k=top_k)
            else:
                reranked = retrieved
            rerank_time = time.perf_counter()
            selected_docs = reranked[: int(config.get("max_docs_per_query", top_k))]
            context = [doc.content for doc, _ in selected_docs]
            answer = generator.generate(query.question, context=context, max_tokens=int(config.get("generation", {}).get("max_tokens", 256)))
            generation_time = time.perf_counter()
            references[query.id] = query.gold_answers
            predictions[query.id] = answer
            retrieval_results[query.id] = [doc.doc_id for doc, _ in reranked]
            payload = {
                "query_id": query.id,
                "question": query.question,
                "retrieved": [
                    {"doc_id": doc.doc_id, "score": float(score)} for doc, score in retrieved
                ],
                "reranked": [
                    {"doc_id": doc.doc_id, "score": float(score)} for doc, score in reranked
                ],
                "selected_context": [doc.doc_id for doc, _ in selected_docs],
                "generated_answer": answer,
                "citations": [doc.doc_id for doc, _ in selected_docs],
                "timing": {
                    "retrieval": retrieval_time - start,
                    "rerank": rerank_time - retrieval_time,
                    "generation": generation_time - rerank_time,
                    "total": generation_time - start,
                },
                "seed": seed,
            }
            handle.write(json.dumps(payload) + "\n")
            latencies.append(payload["timing"])
    metrics = aggregate_retrieval_metrics(
        {query.id: query.gold_passages for query in dataset.queries},
        retrieval_results,
        k_values=config.get("metrics", {}).get("retrieval_cutoffs", [1, 3, 5]),
    )
    generation_metrics = aggregate_generation_metrics(references, predictions)
    return {
        "retrieval": metrics,
        "generation": generation_metrics,
        "latencies": latencies,
        "results_path": results_path,
    }


def run_experiment(config: Config, *, overrides: Sequence[str], repeat: int) -> Path:
    resolved = _resolve_config(config, overrides)
    resolved["repeat"] = repeat
    config_yaml = yaml.safe_dump(resolved, sort_keys=False)
    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    digest = hashlib.sha1(config_yaml.encode("utf-8")).hexdigest()[:8]
    run_dir = Path("runs") / f"{timestamp}_{digest}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.used.yaml").write_text(config_yaml, encoding="utf-8")
    write_environment_snapshot(run_dir / "env.json")
    tracker = Tracker(run_dir)
    tracker.log_params(**{"repeat": repeat, "config_hash": digest})
    retrieval_scores: Dict[str, List[float]] = defaultdict(list)
    generation_scores: Dict[str, List[float]] = defaultdict(list)
    latencies_total: List[float] = []
    combined_results = run_dir / "results_raw.jsonl"
    if combined_results.exists():
        combined_results.unlink()
    for idx in range(repeat):
        seed = int(resolved.get("seed", 42)) + idx
        repeat_dir = run_dir / f"seed_{seed}"
        repeat_dir.mkdir(exist_ok=True)
        result = _run_single(resolved, repeat_dir, seed)
        if result.get("results_path"):
            combined_results.parent.mkdir(exist_ok=True)
            with combined_results.open("a", encoding="utf-8") as dest:
                dest.write(Path(result["results_path"]).read_text(encoding="utf-8"))
        for metric_name, values in result["retrieval"].items():
            if isinstance(values, dict):
                for cutoff, score in values.items():
                    retrieval_scores[f"{metric_name}@{cutoff}"].append(float(score))
            else:
                retrieval_scores[metric_name].append(float(values))
        for metric_name, score in result["generation"].items():
            generation_scores[metric_name].append(float(score))
        latencies_total.extend(item["total"] for item in result["latencies"])
    metrics_path = run_dir / "metrics.json"
    aggregated = {
        "retrieval": {
            name: {"mean": mean_and_std(scores)[0], "std": mean_and_std(scores)[1]}
            for name, scores in retrieval_scores.items()
        },
        "generation": {
            name: {"mean": mean_and_std(scores)[0], "std": mean_and_std(scores)[1]}
            for name, scores in generation_scores.items()
        },
        "latency": {
            "mean": mean_and_std(latencies_total)[0],
            "std": mean_and_std(latencies_total)[1],
        },
    }
    metrics_path.write_text(json.dumps(aggregated, indent=2, sort_keys=True), encoding="utf-8")
    bar_payload: Dict[str, Dict[int, float]] = defaultdict(dict)
    for name, scores in retrieval_scores.items():
        if "@" not in name:
            continue
        metric_name, cutoff = name.split("@", 1)
        try:
            k_value = int(cutoff)
        except ValueError:
            continue
        mean_score = mean_and_std(scores)[0]
        bar_payload[metric_name][k_value] = mean_score
    if bar_payload:
        plot_bar_metrics(bar_payload, title="Retrieval Metrics", output_dir=run_dir / "plots")
    if latencies_total:
        plot_latency(latencies_total, list(range(1, len(latencies_total) + 1)), output_dir=run_dir / "plots")
    recall_scores = retrieval_scores.get("recall@5", [])
    scatter_y = latencies_total[: len(recall_scores)] if recall_scores else []
    if recall_scores and scatter_y:
        plot_scatter(
            recall_scores[: len(scatter_y)],
            scatter_y,
            x_label="Recall@5",
            y_label="Latency",
            title="Coverage vs Accuracy",
            output_dir=run_dir / "plots",
        )
    save_metric_table(retrieval_scores, output_path=run_dir / "tables" / "retrieval.md")
    save_metric_table(generation_scores, output_path=run_dir / "tables" / "generation.md")
    return run_dir


def main(argv: Sequence[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description="Run benchmarking experiments")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--override", action="append", default=[], help="Override config values (key=value)")
    parser.add_argument("--repeat", type=int, default=1)
    args = parser.parse_args(argv)
    config = _load_config(args.config)
    return run_experiment(config, overrides=args.override, repeat=args.repeat)


if __name__ == "__main__":  # pragma: no cover
    main()
