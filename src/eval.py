"""Retrieval evaluation helpers for the FTS contract search system.

This module evaluates ranked retrieval output from ``search_contracts`` using
manual judgments. It focuses on the ranking modes already supported by the
app: ``text``, ``semantic``, and ``hybrid``.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Iterable, Optional, Literal

@dataclass(frozen=True)
class Judgment:
    ocid: str
    relevance: int


@dataclass(frozen=True)
class EvaluationQuery:
    query_id: str
    query: str
    judgments: tuple[Judgment, ...]


@dataclass(frozen=True)
class RagResult:
    """Result from RAG evaluation: answer + cited sources."""
    answer: str
    sources: list[str]  # OCIDs cited in the answer
    parsed_intent: dict = field(default_factory=dict)


def _load_evaluation_queries(path: Path) -> list[EvaluationQuery]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    entries = raw.get("queries", raw)
    if not isinstance(entries, list):
        raise ValueError("Evaluation file must contain a list of queries or a {'queries': [...]} object.")

    queries: list[EvaluationQuery] = []
    for index, entry in enumerate(entries, start=1):
        if not isinstance(entry, dict):
            raise ValueError(f"Query entry #{index} must be an object.")

        query_id = str(entry.get("id") or entry.get("query_id") or f"q{index}")
        query_text = str(entry.get("query") or "").strip()
        if not query_text:
            raise ValueError(f"Query entry #{index} is missing a non-empty 'query' field.")

        judgments = _parse_judgments(entry, query_id)
        queries.append(EvaluationQuery(query_id=query_id, query=query_text, judgments=tuple(judgments)))

    return queries


def _parse_judgments(entry: dict, query_id: str) -> list[Judgment]:
    judgments: list[Judgment] = []

    if isinstance(entry.get("judgments"), list):
        for item in entry["judgments"]:
            if not isinstance(item, dict):
                raise ValueError(f"Query {query_id}: each judgment must be an object.")
            ocid = str(item.get("ocid") or "").strip()
            if not ocid:
                raise ValueError(f"Query {query_id}: a judgment is missing 'ocid'.")
            relevance = int(item.get("relevance", 1))
            judgments.append(Judgment(ocid=ocid, relevance=relevance))
        return judgments

    if isinstance(entry.get("relevant_ocids"), list):
        for ocid in entry["relevant_ocids"]:
            ocid_text = str(ocid).strip()
            if ocid_text:
                judgments.append(Judgment(ocid=ocid_text, relevance=1))
        return judgments

    raise ValueError(
        f"Query {query_id} must define either 'judgments' or 'relevant_ocids' to support evaluation."
    )


def precision_at_k(ranked_ocids: Iterable[str], relevant: set[str], k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    top_k = list(ranked_ocids)[:k]
    if not top_k:
        return 0.0
    return sum(1 for ocid in top_k if ocid in relevant) / k


def recall_at_k(ranked_ocids: Iterable[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    top_k = list(ranked_ocids)[:k]
    return sum(1 for ocid in top_k if ocid in relevant) / len(relevant)


def reciprocal_rank(ranked_ocids: Iterable[str], relevant: set[str]) -> float:
    for position, ocid in enumerate(ranked_ocids, start=1):
        if ocid in relevant:
            return 1.0 / position
    return 0.0


def dcg_at_k(ranked_ocids: Iterable[str], gains: dict[str, int], k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    score = 0.0
    for position, ocid in enumerate(list(ranked_ocids)[:k], start=1):
        gain = gains.get(ocid, 0)
        if gain > 0:
            score += (2**gain - 1) / math.log2(position + 1)
    return score


def ndcg_at_k(ranked_ocids: Iterable[str], gains: dict[str, int], k: int) -> float:
    ideal_order = sorted(gains.items(), key=lambda item: item[1], reverse=True)
    ideal_ranking = [ocid for ocid, gain in ideal_order if gain > 0]
    if not ideal_ranking:
        return 0.0

    dcg = dcg_at_k(ranked_ocids, gains, k)
    ideal_dcg = dcg_at_k(ideal_ranking, gains, k)
    return dcg / ideal_dcg if ideal_dcg else 0.0


def source_coverage(cited_sources: list[str], relevant: set[str]) -> float:
    """Measure what fraction of cited sources are relevant."""
    if not cited_sources:
        return 0.0
    return sum(1 for ocid in cited_sources if ocid in relevant) / len(cited_sources)


def source_precision_at_k(cited_sources: list[str], relevant: set[str], k: int = None) -> float:
    """Precision of cited sources (optionally limited to first k)."""
    if k is None:
        k = len(cited_sources)
    top_k = cited_sources[:k]
    if not top_k:
        return 0.0
    return sum(1 for ocid in top_k if ocid in relevant) / k


def source_recall(cited_sources: list[str], relevant: set[str]) -> float:
    """What fraction of relevant documents were cited."""
    if not relevant:
        return 0.0
    return sum(1 for ocid in cited_sources if ocid in relevant) / len(relevant)


def _extract_ranked_ocids(results: list[dict]) -> list[str]:
    ocids: list[str] = []
    for row in results:
        ocid = row.get("ocid")
        if ocid:
            ocids.append(str(ocid))
    return ocids


def _build_gains(judgments: Iterable[Judgment]) -> dict[str, int]:
    gains: dict[str, int] = {}
    for judgment in judgments:
        current = gains.get(judgment.ocid, 0)
        gains[judgment.ocid] = max(current, int(judgment.relevance))
    return gains


def evaluate_query(es, evaluation_query: EvaluationQuery, search_type: str, k: int) -> dict:
    from src.es_client import search_contracts

    response = search_contracts(
        es=es,
        query_text=evaluation_query.query,
        search_type=search_type,
        size=k,
        page=1,
    )
    results = response.get("results", [])
    ranked_ocids = _extract_ranked_ocids(results)
    gains = _build_gains(evaluation_query.judgments)
    relevant = {judgment.ocid for judgment in evaluation_query.judgments if judgment.relevance > 0}

    return {
        "query_id": evaluation_query.query_id,
        "query": evaluation_query.query,
        "search_type": search_type,
        "k": k,
        "precision_at_k": precision_at_k(ranked_ocids, relevant, k),
        "recall_at_k": recall_at_k(ranked_ocids, relevant, k),
        "mrr": reciprocal_rank(ranked_ocids, relevant),
        "ndcg_at_k": ndcg_at_k(ranked_ocids, gains, k),
        "returned": len(ranked_ocids),
        "top_ocids": ranked_ocids,
        "parsed_intent": response.get("parsed_intent", {}),
    }


def evaluate_rag_query(es, evaluation_query: EvaluationQuery, k: int) -> dict:
    """Evaluate RAG-based answer generation using cited sources."""
    from src.rag import rag_answer

    rag_result = rag_answer(
        query=evaluation_query.query,
        es=es,
        top_k=k,
    )

    cited_sources = rag_result.get("sources", [])
    gains = _build_gains(evaluation_query.judgments)
    relevant = {judgment.ocid for judgment in evaluation_query.judgments if judgment.relevance > 0}

    return {
        "query_id": evaluation_query.query_id,
        "query": evaluation_query.query,
        "search_type": "rag",
        "k": k,
        "answer_snippet": rag_result.get("answer", "")[:200],
        "sources_cited": cited_sources,
        "source_coverage": source_coverage(cited_sources, relevant),
        "source_precision": source_precision_at_k(cited_sources, relevant, k),
        "source_recall": source_recall(cited_sources, relevant),
        "cited_count": len(cited_sources),
        "relevant_count": len(relevant),
        "query_variations_generated": len(rag_result.get("query_variations", [])),
        "total_candidates_evaluated": rag_result.get("total_candidates_evaluated", 0),
        "parsed_intent": rag_result.get("parsed_intent", {}),
    }


def evaluate_dataset(
    evaluation_queries: list[EvaluationQuery],
    search_types: tuple[str, ...] = ("text", "semantic", "hybrid"),
    k: int = 5,
    include_rag: bool = False,
) -> dict:
    from src.es_client import get_es_client

    es = get_es_client()
    per_query: list[dict] = []

    for evaluation_query in evaluation_queries:
        for search_type in search_types:
            per_query.append(evaluate_query(es, evaluation_query, search_type, k))

        if include_rag:
            try:
                per_query.append(evaluate_rag_query(es, evaluation_query, k))
            except Exception as e:
                import logging
                logging.warning(f"RAG evaluation failed for query {evaluation_query.query_id}: {e}")

    summaries = []
    eval_types = list(search_types)
    if include_rag:
        eval_types.append("rag")

    for search_type in eval_types:
        rows = [row for row in per_query if row.get("search_type") == search_type]
        if not rows:
            continue

        summary = {
            "search_type": search_type,
            "queries": len(rows),
            "zero_hit_queries": sum(1 for row in rows if row.get("search_type") != "rag" and not row.get("top_ocids")),
            "intent_relaxations": _count_intent_fallbacks(rows),
        }

        if search_type == "rag":
            summary.update({
                "source_coverage": _mean_or_zero(row["source_coverage"] for row in rows),
                "source_precision": _mean_or_zero(row["source_precision"] for row in rows),
                "source_recall": _mean_or_zero(row["source_recall"] for row in rows),
                "avg_citations": _mean_or_zero(row["cited_count"] for row in rows),
                "avg_query_variations": _mean_or_zero(row["query_variations_generated"] for row in rows),
                "avg_candidates_per_query": _mean_or_zero(row["total_candidates_evaluated"] for row in rows),
            })
        else:
            summary.update({
                "precision_at_k": _mean_or_zero(row["precision_at_k"] for row in rows),
                "recall_at_k": _mean_or_zero(row["recall_at_k"] for row in rows),
                "mrr": _mean_or_zero(row["mrr"] for row in rows),
                "ndcg_at_k": _mean_or_zero(row["ndcg_at_k"] for row in rows),
            })

        summaries.append(summary)

    return {"k": k, "summaries": summaries, "per_query": per_query}


def _mean_or_zero(values: Iterable[float]) -> float:
    collected = list(values)
    return mean(collected) if collected else 0.0


def _count_intent_fallbacks(rows: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        parsed_intent = row.get("parsed_intent") or {}
        fallback = parsed_intent.get("fallback")
        if fallback:
            counts[fallback] = counts.get(fallback, 0) + 1
    return counts


def _format_summary(report: dict) -> str:
    lines = [f"Evaluation Summary @ k={report['k']}\n"]
    lines.append("=" * 80)

    for summary in report["summaries"]:
        search_type = summary["search_type"]
        lines.append(f"\n{search_type.upper()} Search")
        lines.append("-" * 40)

        lines.append(f"  Queries evaluated: {summary['queries']}")

        if search_type == "rag":
            lines.append(f"  Source Coverage:  {summary.get('source_coverage', 0):.3f} (relevant sources / total cited)")
            lines.append(f"  Source Precision: {summary.get('source_precision', 0):.3f} (relevant / cited in top-k)")
            lines.append(f"  Source Recall:    {summary.get('source_recall', 0):.3f} (cited relevant / total relevant)")
            lines.append(f"  Avg Citations:    {summary.get('avg_citations', 0):.1f}")
            lines.append(f"  Query Variations: {summary.get('avg_query_variations', 0):.1f} per query")
            lines.append(f"  Candidates Pool:  {summary.get('avg_candidates_per_query', 0):.0f} docs evaluated")
        else:
            lines.append(f"  Precision@k:      {summary.get('precision_at_k', 0):.3f}")
            lines.append(f"  Recall@k:         {summary.get('recall_at_k', 0):.3f}")
            lines.append(f"  MRR:              {summary.get('mrr', 0):.3f}")
            lines.append(f"  nDCG@k:           {summary.get('ndcg_at_k', 0):.3f}")
            lines.append(f"  Zero-hit queries: {summary.get('zero_hit_queries', 0)}")

        if summary.get("intent_relaxations"):
            fallbacks_str = ", ".join(f"{k}:{v}" for k, v in summary["intent_relaxations"].items())
            lines.append(f"  Intent fallbacks: {fallbacks_str}")

    lines.append("\n" + "=" * 80)
    return "\n".join(lines)


def _format_per_query_details(report: dict) -> str:
    """Format detailed per-query breakdown."""
    lines = ["Per-Query Details\n", "=" * 120]

    # Group by query
    queries_dict: dict[str, list] = {}
    for row in report["per_query"]:
        qid = row["query_id"]
        if qid not in queries_dict:
            queries_dict[qid] = []
        queries_dict[qid].append(row)

    for query_id in sorted(queries_dict.keys()):
        rows = queries_dict[query_id]
        query_text = rows[0].get("query", "N/A")[:80]
        lines.append(f"\nQuery: {query_id} — {query_text}")
        lines.append("-" * 120)

        for row in rows:
            st = row["search_type"]
            if st == "rag":
                lines.append(
                    f"  {st:10s} | cov={row.get('source_coverage', 0):.2f} "
                    f"prec={row.get('source_precision', 0):.2f} "
                    f"recall={row.get('source_recall', 0):.2f} | "
                    f"cited={row.get('cited_count', 0)}/{row.get('relevant_count', 0)} | "
                    f"vars={row.get('query_variations_generated', 0)} cands={row.get('total_candidates_evaluated', 0)}"
                )
            else:
                top_ocids = row.get("top_ocids", [])[:3]
                lines.append(
                    f"  {st:10s} | p@k={row.get('precision_at_k', 0):.2f} "
                    f"r@k={row.get('recall_at_k', 0):.2f} "
                    f"mrr={row.get('mrr', 0):.2f} "
                    f"ndcg={row.get('ndcg_at_k', 0):.2f} | "
                    f"top_3={top_ocids}"
                )

    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality for the FTS search system.")
    parser.add_argument("--queries", required=True, help="Path to a JSON evaluation set.")
    parser.add_argument("--k", type=int, default=5, help="Cutoff for top-k metrics.")
    parser.add_argument(
        "--search-types",
        nargs="+",
        default=["text", "semantic", "hybrid"],
        choices=["text", "semantic", "hybrid"],
        help="Search modes to evaluate.",
    )
    parser.add_argument("--rag", action="store_true", help="Include RAG answer evaluation.")
    parser.add_argument("--output", help="Optional path to write the full JSON report.")
    parser.add_argument("--detailed", action="store_true", help="Print detailed per-query breakdown.")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    evaluation_queries = _load_evaluation_queries(Path(args.queries))
    report = evaluate_dataset(
        evaluation_queries,
        search_types=tuple(args.search_types),
        k=args.k,
        include_rag=args.rag,
    )

    print(_format_summary(report))

    if args.detailed:
        print("\n")
        print(_format_per_query_details(report))

    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())