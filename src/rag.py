import os
import logging
import json
import concurrent.futures
from functools import lru_cache
from typing import List, Optional

from src.es_client import get_es_client, search_contracts, INDEX_NAME, parse_query_intent

logger = logging.getLogger(__name__)


@lru_cache(maxsize=4)
def _load_text_generation_pipeline(model_name: str):
    """Load and cache a text-generation pipeline for repeated RAG use."""
    try:
        from transformers import pipeline

        pipe = pipeline(
            "text-generation",
            model=model_name,
            device_map="auto" if os.environ.get("LOCAL_GPU") else None,
        )

        # Transformers pipelines often keep a default max_length in generation_config.
        # We use max_new_tokens everywhere, so clear the legacy limit to avoid warnings.
        generation_config = getattr(getattr(pipe, "model", None), "generation_config", None)
        if generation_config is not None and getattr(generation_config, "max_length", None) is not None:
            generation_config.max_length = None

        return pipe
    except Exception as exc:
        logger.warning("Unable to load text-generation model '%s': %s", model_name, exc)
        return None


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


def _generate_query_variations(query: str, model_name: Optional[str] = None, intent: Optional[dict] = None) -> List[str]:
    """
    Use an LLM to generate alternative query formulations for the same information need.
    Takes into account structured intent (filters, services, locations) if provided.
    Returns 2-3 query variations plus the original.
    
    For best results with query expansion, set RAG_MODEL to an instruction-tuned model like:
    - mistral-7b-instruct
    - neural-chat-7b
    - phi-2
    - openhermes-2.5
    """
    queries = [query]

    chosen_model = model_name or os.environ.get("QUERY_INTENT_MODEL") or os.environ.get("RAG_MODEL")
    if not chosen_model:
        logger.debug("No LLM available for query expansion, using original query only")
        return queries

    def _extract_json_from_text(text: str) -> Optional[dict]:
        # Prefer the area after an explicit marker
        if "JSON output:" in text:
            text = text.split("JSON output:", 1)[-1]

        # Find the first '{' and attempt to parse a balanced JSON object
        start = text.find("{")
        if start == -1:
            return None

        depth = 0
        for i in range(start, len(text)):
            ch = text[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : i + 1]
                    try:
                        return json.loads(candidate)
                    except Exception:
                        pass
        return None

    def _looks_like_query(text: str, original: str) -> bool:
        """Check if text looks like a real query (not gibberish, URL, or prompt text)."""
        text_lower = text.lower().strip()
        original_lower = original.lower()
        
        # Reject duplicates, URLs, or very short fragments
        if text_lower == original_lower or len(text) < 5 or "http" in text_lower or "github" in text_lower:
            return False
        
        # Reject if it's mostly one word repeated or looks like corrupted text
        if text.count("  ") > 2 or text.count("\\n") > 1:
            return False
        
        # Reject highly repetitive text (e.g., "SELECT FROM FROM FROM...")
        if text.count(" FROM ") > 3 or text.count(" from ") > 3:
            return False
        if text.count('""') > 2 or text.count("''") > 2:
            return False
        
        # Reject known prompt instruction phrases
        prompt_phrases = {
            "given this", "generate", "alternative phrasing", "search for the same",
            "from different angles", "synonyms", "broader", "narrower", "keywords",
            "ensure each variation", "preserves the key concepts", "return only",
            "json object", "variations key", "original query", "json output"
        }
        
        for phrase in prompt_phrases:
            if phrase in text_lower:
                return False
        
        # Should have reasonable word count (at least 3 words for a proper query)
        words = text_lower.split()
        if len(words) < 3:
            return False
        
        # Reject SQL-like or code-like patterns
        if text_lower.startswith("select") or "select " in text_lower[:50]:
            return False
        
        return True

    try:
        pipe = _load_text_generation_pipeline(chosen_model)
        if pipe is None:
            return queries

        # Build intent context for the prompt
        intent_context = ""
        if intent:
            context_parts = []
            if intent.get("filters"):
                filters_str = ", ".join(f"{k}: {v}" for k, v in intent["filters"].items())
                context_parts.append(f"Filters: {filters_str}")
            if intent.get("services"):
                services_str = ", ".join(intent["services"])
                context_parts.append(f"Services: {services_str}")
            if intent.get("locations"):
                locations_str = ", ".join(intent["locations"])
                context_parts.append(f"Locations: {locations_str}")
            if context_parts:
                intent_context = f"\n\nStructured intent extracted from query:\n" + "\n".join(context_parts)

        expansion_prompt = (
            f"Given this procurement search query, generate 2-3 alternative phrasings that search for the same thing "
            f"but from different angles (e.g., synonyms, broader/narrower terms, different keywords). "
            f"Ensure each variation preserves the key concepts and constraints from the original query.{intent_context}\n\n"
            f"Return ONLY a JSON object with an 'variations' key containing a list of strings.\n\n"
            f"Original query: {query}\n\n"
            f"JSON output:"
        )

        # Try to request multiple sampled outputs to get diverse variations
        try:
            outputs = pipe(
                expansion_prompt,
                max_new_tokens=200,
                do_sample=True,
                temperature=0.7,
                top_k=50,
                num_return_sequences=3,
            )
        except TypeError:
            # Some pipeline/backends don't accept num_return_sequences; fall back to single sampled output
            outputs = [pipe(expansion_prompt, max_new_tokens=200, do_sample=True, temperature=0.7, top_k=50)[0]]

        parsed_any = False
        for out in outputs:
            response_text = out.get("generated_text") if isinstance(out, dict) else str(out)
            logger.debug("Query expansion response: %s", response_text[:300])
            parsed = _extract_json_from_text(response_text)
            if parsed and isinstance(parsed.get("variations"), list):
                for q in parsed["variations"]:
                    if not q:
                        continue
                    q_str = q.strip()
                    if q_str and _looks_like_query(q_str, query) and q_str not in queries:
                        queries.append(q_str)
                parsed_any = True

        if not parsed_any:
            logger.debug("No valid JSON variations parsed from LLM outputs.")
            # Only attempt line-based fallback if response looks reasonable
            try:
                fallback_text = outputs[0].get("generated_text") if isinstance(outputs[0], dict) else str(outputs[0])
                # Extract only clean lines that don't look like prompt echo or gibberish
                for line in fallback_text.splitlines():
                    li = line.strip(" -\n\r\t\"'")
                    if _looks_like_query(li, query) and li not in queries:
                        queries.append(li)
                        if len(queries) >= 4:  # Stop after finding 3 variations
                            break
            except Exception as e:
                logger.debug("Fallback line-filtering failed: %s", e)

    except Exception as e:
        logger.warning(f"Query expansion failed: {e}. Tip: For better results, set RAG_MODEL to an instruction-tuned model.")

    return queries[:4]  # Limit to 4 total queries (original + 3 variations)


def _score_result(result: dict, rank_position: int, total_search_count: int) -> float:
    """
    Score a result based on:
    - Relevance score from ES (if available)
    - Rank position (earlier is better)
    - Presence of key fields
    """
    score = 0.0
    
    # Base score from ES relevance (normalized to 0-1 range)
    es_score = result.get("relevance_score", 0.0)
    if isinstance(es_score, (int, float)):
        score += min(es_score / 10.0, 1.0) * 0.5  # Weight: 50%
    
    # Rank bonus: earlier rank = higher score
    rank_bonus = max(0, 1.0 - (rank_position / 100.0)) * 0.3  # Weight: 30%
    score += rank_bonus
    
    # Presence of important fields
    field_bonus = 0.0
    if result.get("title"):
        field_bonus += 0.05
    if result.get("description") or result.get("chunk_text"):
        field_bonus += 0.05
    if result.get("buyer_name"):
        field_bonus += 0.05
    if result.get("value_amount"):
        field_bonus += 0.05
    score += field_bonus  # Weight: 20%
    
    return score


def _merge_and_rank_results(all_results: List[dict], top_k: int) -> List[dict]:
    """
    Merge results from multiple searches, deduplicate by OCID, score by rank position,
    and return top-k unique documents.
    """
    # Deduplicate by a canonical contract identifier when available (contract_id),
    # falling back to notice_id and then ocid. Keep the highest-scored version.
    canonical_map = {}
    for rank, result in enumerate(all_results):
        canonical_id = (
            result.get("contract_id") or result.get("notice_id") or result.get("ocid")
        )
        if not canonical_id:
            continue

        score = _score_result(result, rank, len(all_results))

        if canonical_id not in canonical_map or score > canonical_map[canonical_id].get("_merge_score", 0):
            result_with_score = dict(result)
            result_with_score["_merge_score"] = score
            canonical_map[canonical_id] = result_with_score

    # Sort by merge score and return top-k
    ranked = sorted(canonical_map.values(), key=lambda x: x.get("_merge_score", 0), reverse=True)
    return ranked[:top_k]


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
    num_candidates: int = 20,
    expand_queries: bool = True,
):
    """
    Run an improved RAG pipeline with query expansion and result ranking.

    - Generates 2-3 query variations if expand_queries=True
    - Runs searches in parallel for all query variations
    - Merges and ranks results by relevance and position
    - Returns top-k unique documents
    
    Args:
        query: The original search query
        es: Elasticsearch client (optional, will create if not provided)
        top_k: Number of final results to return (default 5)
        model_name: Optional LLM name for answer generation and query expansion
        max_length: Max tokens for LLM answer generation
        num_candidates: Number of candidates to retrieve per query before ranking (default 20)
        expand_queries: Whether to generate query variations (default True)
    """
    if es is None:
        es = get_es_client()

    # Parse original query intent
    original_intent = parse_query_intent(query)
    
    # Generate query variations informed by the parsed intent
    queries_to_search = _generate_query_variations(query, model_name, intent=original_intent) if expand_queries else [query]
    logger.info(f"Generated {len(queries_to_search)} query variations for '{query}'")
    
    # Parse intent for each query variation
    query_intents = {q: parse_query_intent(q) for q in queries_to_search}

    # Run searches in parallel
    all_results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(4, len(queries_to_search))) as executor:
        futures = {
            executor.submit(
                search_contracts,
                es=es,
                query_text=q,
                search_type="hybrid",
                size=num_candidates,
                page=1,
            ): q
            for q in queries_to_search
        }

        for future in concurrent.futures.as_completed(futures):
            try:
                resp = future.result()
                results = resp.get("results", [])
                all_results.extend(results)
                logger.debug(f"Retrieved {len(results)} results for query variation")
            except Exception as e:
                logger.warning(f"Search failed for a query variation: {e}")

    if not all_results:
        return {
            "answer": "No matching contracts found.",
            "sources": [],
            "parsed_intent": original_intent,
            "query_variations": queries_to_search,
            "total_candidates_evaluated": 0,
            "workflow": {
                "query": query,
                "original_intent": original_intent,
                "expanded": expand_queries,
                "query_variations": queries_to_search,
                "query_intents": query_intents,
                "searches_run": len(queries_to_search),
                "candidates_per_query": num_candidates,
                "total_candidates_evaluated": 0,
                "ranked_results": [],
                "summary": "No matching contracts were found.",
            },
        }

    # Merge, deduplicate, and rank results
    ranked_results = _merge_and_rank_results(all_results, top_k)

    workflow = {
        "query": query,
        "original_intent": original_intent,
        "expanded": expand_queries,
        "query_variations": queries_to_search,
        "query_intents": query_intents,
        "searches_run": len(queries_to_search),
        "candidates_per_query": num_candidates,
        "total_candidates_evaluated": len(all_results),
        "ranked_results": [
            {
                "rank": index,
                "ocid": result.get("ocid"),
                "title": result.get("title") or "Untitled",
                "buyer_name": result.get("buyer_name") or "Unknown",
            }
            for index, result in enumerate(ranked_results, start=1)
        ],
    }
    
    # Build context from top results
    context = _build_context(ranked_results)
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
            pipe = _load_text_generation_pipeline(model_name)
            if pipe is None:
                raise RuntimeError(f"Could not load text-generation pipeline for {model_name}")

            # Use text-generation pipeline; model_name can be a local path or HF repo id
            out = pipe(prompt, max_new_tokens=max_length, do_sample=False)
            text = out[0]["generated_text"]
            # Trim prompt echo if present
            if prompt in text:
                text = text.split(prompt, 1)[-1].strip()
            return {
                "answer": text,
                "sources": [r.get("ocid") for r in ranked_results],
                "parsed_intent": original_intent,
                "query_variations": queries_to_search,
                "total_candidates_evaluated": len(all_results),
                "workflow": workflow,
            }
        except Exception as e:
            logger.warning("transformers pipeline unavailable or failed: %s", e)

    # Fallback: synthesise a short answer from top titles
    summary_lines = [f"[{i+1}] {r.get('title') or r.get('ocid')} — {r.get('buyer_name') or 'Unknown'}" for i, r in enumerate(ranked_results)]
    answer = "Found the following relevant contracts:\n" + "\n".join(summary_lines)
    return {
        "answer": answer,
        "sources": [r.get("ocid") for r in ranked_results],
        "parsed_intent": original_intent,
        "query_variations": queries_to_search,
        "total_candidates_evaluated": len(all_results),
        "workflow": workflow,
    }


if __name__ == "__main__":
    # quick smoke test if executed directly
    es = get_es_client()
    q = "Find framework agreements for IT services in the NHS"
    print(rag_answer(q, es=es, top_k=3, model_name=os.environ.get("RAG_MODEL")))
