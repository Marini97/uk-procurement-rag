import os
import logging
from typing import List, Optional

from src.es_client import get_es_client, search_contracts, INDEX_NAME

logger = logging.getLogger(__name__)


def _flatten_mapping_fields(properties: dict, prefix: str = "") -> List[tuple[str, str]]:
    """Flatten Elasticsearch mapping properties into (field_path, field_type)."""
    fields: List[tuple[str, str]] = []
    for name, meta in (properties or {}).items():
        path = f"{prefix}.{name}" if prefix else name
        field_type = meta.get("type", "object")
        fields.append((path, field_type))

        # Include multi-fields (e.g. buyer_name.text)
        for sub_name, sub_meta in (meta.get("fields") or {}).items():
            sub_path = f"{path}.{sub_name}"
            sub_type = sub_meta.get("type", "unknown")
            fields.append((sub_path, sub_type))

        # Recurse nested/object properties
        if "properties" in meta:
            fields.extend(_flatten_mapping_fields(meta.get("properties") or {}, path))

    return fields


def _mapping_to_filter_hints(field_type: str) -> str:
    """Suggest filter strategy by ES field type for prompt guidance."""
    keyword_like = {"keyword", "boolean", "integer", "long", "short", "byte", "date"}
    range_like = {"double", "float", "half_float", "scaled_float", "integer", "long", "date"}

    if field_type in keyword_like and field_type in range_like:
        return "term/range"
    if field_type in keyword_like:
        return "term"
    if field_type in range_like:
        return "range"
    if field_type in {"text"}:
        return "match"
    if field_type in {"nested", "object"}:
        return "nested/object"
    return "query-dependent"


def _build_index_schema_context(es, index_name: str) -> str:
    """Fetch index mapping and render all indexed fields with filter hints."""
    try:
        mapping = es.indices.get_mapping(index=index_name)
        properties = (mapping.get(index_name, {}).get("mappings", {}).get("properties", {}))
        fields = _flatten_mapping_fields(properties)
        if not fields:
            return "Index fields unavailable."

        lines = ["Available index fields (field_path: type -> suggested_filter):"]
        for field_path, field_type in sorted(fields, key=lambda x: x[0]):
            lines.append(f"- {field_path}: {field_type} -> {_mapping_to_filter_hints(field_type)}")
        return "\n".join(lines)
    except Exception as e:
        logger.warning("Could not load index mapping for prompt context: %s", e)
        return "Index fields unavailable (mapping lookup failed)."


def _build_context(results: List[dict]) -> str:
    parts = []
    for i, r in enumerate(results, start=1):
        parts.append(
            f"[{i}] Title: {r.get('title') or 'Untitled'}\nBuyer: {r.get('buyer_name') or 'Unknown'}\nOCID: {r.get('ocid') or 'n/a'}\nStatus: {r.get('contract_status') or 'Unknown'}\nValue: {r.get('value_amount') or 'n/a'} {r.get('value_currency') or ''}\nDescription: { (r.get('chunk_text') or r.get('description') or '')[:800]}"
        )
    return "\n\n".join(parts)


def rag_answer(
    query: str,
    es=None,
    top_k: int = 5,
    model_name: Optional[str] = None,
    max_length: int = 256,
):
    """
    Run a simple RAG pipeline: fetch top-k documents and ask a small LLM to answer.

    - If `model_name` is provided and the `transformers` library is available, it will use
      Hugging Face Transformers to generate an answer locally.
    - Otherwise falls back to a simple synthesized answer using the top documents.
    """
    if es is None:
        es = get_es_client()

    # Retrieve top documents using existing search; use hybrid to get good candidates
    resp = search_contracts(es=es, query_text=query, search_type="hybrid", size=top_k)
    results = resp.get("results", [])

    if not results:
        return {"answer": "No matching contracts found.", "sources": []}

    context = _build_context(results)
    schema_context = _build_index_schema_context(es, INDEX_NAME)

    prompt = (
        "You are an expert assistant for UK procurement data. Use both contextual documents and index schema metadata to answer the question.\n\n"
        "When suggesting or applying filters, ONLY use fields listed in the index schema section and choose filter style based on suggested_filter.\n\n"
        "Index schema:\n===\n"
        f"{schema_context}\n===\n\n"
        "Context documents:\n===\n"
        f"{context}\n===\n"
        f"Question: {query}\n\nAnswer concisely, cite documents by number when relevant."
    )

    # Try to use transformers if available and a model name provided
    if model_name:
        try:
            from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

            # Use text-generation pipeline; model_name can be a local path or HF repo id
            pipe = pipeline(
                "text-generation",
                model=model_name,
                device_map="auto" if os.environ.get("LOCAL_GPU") else None,
            )
            out = pipe(prompt, max_new_tokens=max_length, do_sample=False)
            text = out[0]["generated_text"]
            # Trim prompt echo if present
            if prompt in text:
                text = text.split(prompt, 1)[-1].strip()
            return {"answer": text, "sources": [r.get("ocid") for r in results], "parsed_intent": resp.get("parsed_intent")}
        except Exception as e:
            logger.warning("transformers pipeline unavailable or failed: %s", e)

    # Fallback: synthesise a short answer from top titles
    summary_lines = [f"[{i+1}] {r.get('title') or r.get('ocid')} — {r.get('buyer_name') or 'Unknown'}" for i, r in enumerate(results)]
    answer = "Found the following relevant contracts:\n" + "\n".join(summary_lines)
    return {"answer": answer, "sources": [r.get("ocid") for r in results], "parsed_intent": resp.get("parsed_intent")}


if __name__ == "__main__":
    # quick smoke test if executed directly
    es = get_es_client()
    q = "Find framework agreements for IT services in the NHS"
    print(rag_answer(q, es=es, top_k=3, model_name=os.environ.get("RAG_MODEL")))
