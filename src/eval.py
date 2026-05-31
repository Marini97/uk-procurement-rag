"""Retrieval evaluation helpers for the FTS contract search system.

This module evaluates ranked retrieval output from ``search_contracts`` using
manual judgments. It focuses on the ranking modes already supported by the
app: ``text``, ``semantic``, and ``hybrid``.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Iterable, Optional

@dataclass(frozen=True)
class Judgment:
    ocid: str
    relevance: int


@dataclass(frozen=True)
class EvaluationQuery:
    query_id: str
    query: str
    judgments: tuple[Judgment, ...]


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


def evaluate_dataset(
    evaluation_queries: list[EvaluationQuery],
    search_types: tuple[str, ...] = ("text", "semantic", "hybrid"),
    k: int = 5,
) -> dict:
    from src.es_client import get_es_client

    es = get_es_client()
    per_query: list[dict] = []

    for search_type in search_types:
        for evaluation_query in evaluation_queries:
            per_query.append(evaluate_query(es, evaluation_query, search_type, k))

    summaries = []
    for search_type in search_types:
        rows = [row for row in per_query if row["search_type"] == search_type]
        summaries.append(
            {
                "search_type": search_type,
                "queries": len(rows),
                "precision_at_k": _mean_or_zero(row["precision_at_k"] for row in rows),
                "recall_at_k": _mean_or_zero(row["recall_at_k"] for row in rows),
                "mrr": _mean_or_zero(row["mrr"] for row in rows),
                "ndcg_at_k": _mean_or_zero(row["ndcg_at_k"] for row in rows),
                "zero_hit_queries": sum(1 for row in rows if not row["top_ocids"]),
                "intent_relaxations": _count_intent_fallbacks(rows),
            }
        )

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
    lines = [f"Evaluation summary @ k={report['k']}\n"]
    for summary in report["summaries"]:
        lines.append(f"{summary['search_type']}: ")
        lines.append(f"  queries={summary['queries']}")
        lines.append(f"  precision@k={summary['precision_at_k']:.3f}")
        lines.append(f"  recall@k={summary['recall_at_k']:.3f}")
        lines.append(f"  mrr={summary['mrr']:.3f}")
        lines.append(f"  ndcg@k={summary['ndcg_at_k']:.3f}")
        lines.append(f"  zero_hit_queries={summary['zero_hit_queries']}")
        if summary["intent_relaxations"]:
            lines.append(f"  intent_relaxations={summary['intent_relaxations']}")
        lines.append("")
    return "\n".join(lines).rstrip()


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
    parser.add_argument("--output", help="Optional path to write the full JSON report.")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    evaluation_queries = _load_evaluation_queries(Path(args.queries))
    report = evaluate_dataset(
        evaluation_queries,
        search_types=tuple(args.search_types),
        k=args.k,
    )

    print(_format_summary(report))

    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())