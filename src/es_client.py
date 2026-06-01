import logging
import os
import json
import re
from functools import lru_cache
import spacy
import torch
from typing import Optional, Any
from elasticsearch import Elasticsearch, helpers
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

INDEX_NAME = "fts_contracts"
EMBEDDING_DIM = 384  # all-MiniLM-L6-v2

embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cuda" if torch.cuda.is_available() else "cpu")

nlp = spacy.load("en_core_web_sm")


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

def get_es_client() -> Elasticsearch:
    return Elasticsearch(
        "http://localhost:9200",
        request_timeout=60,
        max_retries=3,
        retry_on_timeout=True,
    )


# ---------------------------------------------------------------------------
# Index setup
# ---------------------------------------------------------------------------

def setup_index(es: Elasticsearch, recreate: bool = False) -> None:
    """
    Creates the Elasticsearch index with mappings for hybrid search.
    Set recreate=True to drop and rebuild the index from scratch.
    """
    exists = es.indices.exists(index=INDEX_NAME)
    if exists:
        if recreate:
            logger.info(f"Deleting existing index: {INDEX_NAME}")
            es.indices.delete(index=INDEX_NAME)
        else:
            logger.info(f"Index {INDEX_NAME} already exists. Skipping creation.")
            return

    mapping = {
        "settings": {
            # Disabled during bulk ingest for speed.
            # Call restore_index_settings() after ingestion is complete.
            "refresh_interval": "-1",
            "number_of_replicas": 0,
        },
        "mappings": {
            "properties": {
                "ocid":         {"type": "keyword"},
                "notice_id":    {"type": "keyword"},
                "release_date": {"type": "date"},
                "tender_id":    {"type": "keyword"},
                "title": {
                    "type": "text",
                    "analyzer": "english",
                    "fields": {"keyword": {"type": "keyword", "ignore_above": 512}},
                },
                "description":  {"type": "text", "analyzer": "english"},
                "procurement_method": {"type": "keyword"},
                "buyer_name": {
                    "type": "keyword",
                    "fields": {"text": {"type": "text", "analyzer": "english"}},
                },
                "supplier_names": {
                    "type": "keyword",
                    "fields": {"text": {"type": "text", "analyzer": "english"}},
                },
                "contract_id":     {"type": "keyword"},
                "contract_status": {"type": "keyword"},
                "value_amount":    {"type": "double"},
                "value_currency":  {"type": "keyword"},
                "cpv_codes":       {"type": "keyword"},
                "cpv_descriptions": {"type": "text", "analyzer": "english"},
                "chunk_text":      {"type": "text", "analyzer": "english"},
                "embedding": {
                    "type": "dense_vector",
                    "dims": EMBEDDING_DIM,
                    "index": True,
                    "similarity": "cosine",
                    "index_options": {
                        "type": "hnsw",
                        "m": 16,
                        "ef_construction": 100,
                    },
                },

                "url": {"type": "keyword"},
                "documents": {"type": "text", "analyzer": "english"},

                "buyer_id": {"type": "keyword"},
                "buyer_contact": {"type": "text", "analyzer": "english"},
                "buyer_address": {"type": "text", "analyzer": "english"},
                "buyer_region": {"type": "keyword"},
                "buyer_country": {"type": "keyword"},

                "supplier_ids": {"type": "keyword"},
                "supplier_names_text": {"type": "text", "analyzer": "english"},
                "supplier_countries": {"type": "keyword"},
                "supplier_count": {"type": "integer"},

                "award_id": {"type": "keyword"},
                "award_date": {"type": "date"},
                "award_value_amount": {"type": "double"},
                "award_value_currency": {"type": "keyword"},

                "tender_period_start": {"type": "date"},
                "tender_period_end": {"type": "date"},

                "lots": {
                    "type": "nested",
                    "properties": {
                        "id": {"type": "keyword"},
                        "title": {"type": "text", "analyzer": "english"},
                        "value_amount": {"type": "double"},
                        "value_currency": {"type": "keyword"},
                    },
                },
                "items": {"type": "text", "analyzer": "english"},

                "classification_codes": {"type": "keyword"},
                "procurement_category": {"type": "keyword"},
                "is_framework": {"type": "boolean"},
                "framework_duration_months": {"type": "integer"},
                "notes": {"type": "text", "analyzer": "english"},
            }
        }
    }

    es.indices.create(index=INDEX_NAME, body=mapping)
    logger.info(f"Created index: {INDEX_NAME}")


def restore_index_settings(es: Elasticsearch) -> None:
    """
    Re-enables refresh and replicas after bulk ingestion.
    Call this once your ingest pipeline is complete.
    """
    es.indices.put_settings(
        index=INDEX_NAME,
        body={"refresh_interval": "1s", "number_of_replicas": 1},
    )
    es.indices.forcemerge(index=INDEX_NAME, max_num_segments=5)
    es.indices.refresh(index=INDEX_NAME)
    logger.info("Restored refresh interval and replicas. Force-merge triggered.")


# ---------------------------------------------------------------------------
# Bulk indexing
# ---------------------------------------------------------------------------

def bulk_index_docs(es: Elasticsearch, documents: list) -> int:
    """
    Bulk indexes a list of normalised contract documents.
    Each document may contain a top-level '_id' key for deterministic IDs.
    Returns the number of successfully indexed documents.
    """
    actions = []
    for doc in documents:
        doc = dict(doc)
        action = {"_index": INDEX_NAME, "_source": doc}
        if "_id" in doc:
            action["_id"] = doc.pop("_id")
        actions.append(action)

    success, errors = helpers.bulk(
        es,
        actions,
        stats_only=False,
        raise_on_error=False,
        chunk_size=500,
        request_timeout=120,
    )
    for err in errors[:5]:
        logger.error(f"Bulk index error: {err}")
    logger.info(f"Bulk index: {success} documents indexed successfully")
    return success


INTENT_FILTER_KEYS = {"procurement_method", "buyer_name", "contract_status", "is_framework"}


@lru_cache(maxsize=4)
def _load_intent_pipeline(model_name: str):
    """Load and cache a text-generation pipeline for intent extraction."""
    try:
        from transformers import pipeline

        return pipeline(
            "text-generation",
            model=model_name,
            device_map="auto" if torch.cuda.is_available() and os.environ.get("LOCAL_GPU") else None,
        )
    except Exception as exc:
        logger.warning("Unable to load intent model '%s': %s", model_name, exc)
        return None


def _extract_json_block(text: str) -> dict[str, Any] | None:
    """Best-effort extraction of the first JSON object from model output."""
    if not text:
        return None

    candidates = []
    fenced = re.findall(r"```json\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    candidates.extend(fenced)
    brace_match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if brace_match:
        candidates.append(brace_match.group(0))

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            continue
    return None


def _normalize_intent_payload(payload: dict[str, Any], query_text: str) -> dict:
    """Normalize model output into the intent schema expected by search_contracts."""
    filters = payload.get("filters") or {}
    normalized_filters = {k: v for k, v in filters.items() if k in INTENT_FILTER_KEYS and v not in (None, "")}

    services = payload.get("services") or []
    locations = payload.get("locations") or []
    clean_text = payload.get("clean_text") or query_text

    if isinstance(services, str):
        services = [services]
    if isinstance(locations, str):
        locations = [locations]

    return {
        "raw": payload.get("raw") or query_text,
        "filters": normalized_filters,
        "locations": [loc for loc in locations if loc],
        "services": [svc for svc in services if svc],
        "clean_text": clean_text,
    }


def _extract_intent_with_llm(query_text: str, model_name: Optional[str] = None) -> dict | None:
    """Ask a text-generation model to turn a query into structured procurement intent."""
    chosen_model = model_name or os.environ.get("QUERY_INTENT_MODEL") or os.environ.get("RAG_MODEL")
    if not chosen_model:
        return None

    pipe = _load_intent_pipeline(chosen_model)
    if pipe is None:
        return None

    prompt = (
        "Extract procurement search intent from the user query. Return ONLY valid JSON. "
        "Use this schema: {"
        '"raw": string, '
        '"filters": {"procurement_method"?: string, "buyer_name"?: string, '
        '"contract_status"?: string, "is_framework"?: boolean}, '
        '"locations": string[], "services": string[], "clean_text": string' 
        "}. "
        "If a field is not present, omit it or leave it empty. "
        "Infer framework, buyer, contract status, service type, and location from meaning, not keyword rules. "
        f"Query: {query_text}"
    )

    try:
        output = pipe(prompt, max_new_tokens=220, do_sample=False, return_full_text=False)
        generated = output[0].get("generated_text", "") if output else ""
        payload = _extract_json_block(generated)
        if not payload:
            return None
        return _normalize_intent_payload(payload, query_text)
    except Exception as exc:
        logger.warning("Intent LLM parsing failed: %s", exc)
        return None


def _fallback_intent_parse(query_text: str) -> dict:
    """Fallback parser when no model is available; relies on spaCy NER only."""
    normalized = (query_text or "").replace("-", " ")
    intent = {
        "raw": query_text,
        "filters": {},
        "locations": [],
        "services": [],
        "clean_text": query_text or "",
    }

    if nlp is None:
        return intent

    doc = nlp(normalized)
    for ent in doc.ents:
        if ent.label_ == "ORG" and "buyer_name" not in intent["filters"]:
            intent["filters"]["buyer_name"] = ent.text
        elif ent.label_ in ("GPE", "LOC"):
            intent["locations"].append(ent.text)

    # If the model is unavailable, preserve the raw query as clean_text.
    # The semantic and BM25 legs still work against the original query.
    return intent


# ---------------------------------------------------------------------------
# Query intent parsing
# ---------------------------------------------------------------------------

def parse_query_intent(query_text: str) -> dict:
    """
    Extracts structured intent from a free-text procurement query.

    Returns:
        raw         - original query string
        filters     - structured field filters (procurement_method, buyer_name)
        locations   - GPE entities found (used for BM25 boosting)
        clean_text  - query with procurement keywords stripped (used for embeddings)
    """
    intent = _extract_intent_with_llm(query_text)
    if intent is not None:
        return intent

    intent = _fallback_intent_parse(query_text)

    # Light cleanup for the fallback path only: keep embeddings on the original query.
    # This avoids relying on keyword dictionaries while still extracting useful structure.
    if not intent.get("locations"):
        m = re.search(r"\b(?:in|near)\s+([A-Za-z\-]+(?:\s+[A-Za-z\-]+){0,2})", query_text or "")
        if m:
            intent["locations"].append(m.group(1).strip().title())

    return intent


# ---------------------------------------------------------------------------
# Client-side Reciprocal Rank Fusion (free alternative to ES-native RRF)
# ---------------------------------------------------------------------------

def reciprocal_rank_fusion(
    result_lists: list[list[dict]],
    k: int = 60,
    id_field: str = "_id",
) -> list[dict]:
    """
    Merges multiple ranked result lists using Reciprocal Rank Fusion (RRF).

    RRF score for document d = sum over each list of 1 / (k + rank(d))

    This avoids needing to normalise BM25 and cosine scores onto the same scale —
    only rank positions matter. Identical to the server-side Elasticsearch RRF
    retriever in behaviour, but runs in Python and works on any licence tier.

    Parameters
    ----------
    result_lists : list of hit lists, each from a separate ES search response
    k            : rank constant (default 60, same as ES default)
    id_field     : the field used to identify unique documents across lists
    """
    scores: dict[str, float] = {}
    docs:   dict[str, dict]  = {}

    for result_list in result_lists:
        for rank, hit in enumerate(result_list, start=1):
            doc_id = hit[id_field]
            rrf_score = 1.0 / (k + rank)
            scores[doc_id] = scores.get(doc_id, 0.0) + rrf_score
            if doc_id not in docs:
                docs[doc_id] = hit

    # Sort descending by accumulated RRF score
    sorted_ids = sorted(scores, key=lambda d: scores[d], reverse=True)
    fused = []
    for doc_id in sorted_ids:
        hit = docs[doc_id]
        hit["_rrf_score"] = round(scores[doc_id], 6)
        fused.append(hit)

    return fused


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def search_contracts(
    es: Elasticsearch,
    query_text: Optional[str] = None,
    procurement_method: Optional[str] = None,
    contract_status: Optional[str] = None,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    search_type: str = "hybrid",
    sort: str = "relevance",
    page: int = 1,
    size: int = 20,
    use_intent_parsing: bool = True,
    rrf_k: int = 60,
) -> dict:
    """
    Search contracts using BM25, semantic (kNN), or hybrid (client-side RRF) search.

    Parameters
    ----------
    query_text          : free-text query
    procurement_method  : exact keyword filter (overrides intent detection)
    contract_status     : exact keyword filter
    min_value/max_value : value_amount range filters
    search_type         : "text" | "semantic" | "hybrid"
    page / size         : pagination
    use_intent_parsing  : parse procurement intent from query_text via spaCy
    rrf_k               : RRF rank constant (default 60)
    """
    # --- Intent parsing ---
    intent = {
        "raw": query_text,
        "filters": {},
        "locations": [],
        "clean_text": query_text or "",
    }
    if query_text and use_intent_parsing:
        intent = parse_query_intent(query_text)

    # Explicit args override intent-detected filters
    if procurement_method:
        intent["filters"]["procurement_method"] = procurement_method
    if contract_status:
        intent["filters"]["contract_status"] = contract_status

    # --- Build filter clauses ---
    filters = []
    if intent["filters"].get("procurement_method"):
        filters.append({"term": {"procurement_method": intent["filters"]["procurement_method"]}})
    if intent["filters"].get("contract_status"):
        filters.append({"term": {"contract_status": intent["filters"]["contract_status"]}})
    if intent["filters"].get("buyer_name"):
        filters.append({"match": {"buyer_name.text": intent["filters"]["buyer_name"]}})
    if intent["filters"].get("is_framework"):
        # boolean flag in index: filter for true frameworks when detected
        filters.append({"term": {"is_framework": True}})
    if min_value is not None or max_value is not None:
        range_q = {}
        if min_value is not None:
            range_q["gte"] = min_value
        if max_value is not None:
            range_q["lte"] = max_value
        filters.append({"range": {"value_amount": range_q}})

    # Location entities boost relevance in BM25 leg
    location_boosts = [
        {
            "multi_match": {
                "query": loc,
                "fields": ["chunk_text", "description", "buyer_name.text", "cpv_descriptions", "cpv_codes"],
                "boost": 1.5,
            }
        }
        for loc in intent.get("locations", [])
    ]

    embed_text = intent.get("clean_text") or query_text or ""

    # Build service/category boosts for BM25 when the user mentions service types
    service_boosts = []
    for svc in intent.get("services", []) or []:
        service_boosts.append({
            "match_phrase": {"chunk_text": {"query": svc, "boost": 2.0}}
        })
        service_boosts.append({
            "match_phrase": {"title": {"query": svc, "boost": 2.0}}
        })

    # Empty query should behave like a browsable listing with real totals.
    # This avoids hybrid candidate caps (e.g. 50) from truncating the homepage.
    if not embed_text.strip():
        hits, total = _browse_search(es, filters, page, size, sort=sort)
        aggs = _run_aggregations(es, filters)
        return _format_response(hits, total, aggs, intent)

    # --- Dispatch to search strategy ---
    if search_type == "text":
        hits = _text_search(es, embed_text, filters, location_boosts, size, service_boosts=service_boosts, sort=sort)
        total = len(hits)

    elif search_type == "semantic":
        query_vector = embedder.encode(embed_text, normalize_embeddings=True).tolist()
        hits = _semantic_search(es, query_vector, filters, size)
        # apply client-side sort for semantic results if requested
        if sort and sort != "relevance":
            hits = _sort_hits(hits, sort)
        total = len(hits)

    else:  # hybrid — two queries + client-side RRF
        query_vector = embedder.encode(embed_text, normalize_embeddings=True).tolist()

        # Fetch more candidates per leg so RRF has enough to work with
        fetch_size = max(size * 3, 50)

        bm25_hits = _text_search(es, embed_text, filters, location_boosts, fetch_size, service_boosts=service_boosts, sort=sort)
        knn_hits  = _semantic_search(es, query_vector, filters, fetch_size)

        fused = reciprocal_rank_fusion([bm25_hits, knn_hits], k=rrf_k)
        # apply post-fusion sort if requested (non-relevance)
        if sort and sort != "relevance":
            fused = _sort_hits(fused, sort)

        hits  = fused[:size]
        total = len(fused)

    # Aggregations (separate lightweight query)
    aggs = _run_aggregations(es, filters)

    # Paginate for non-hybrid (hybrid is already sliced above)
    if search_type != "hybrid":
        from_ = (page - 1) * size
        hits = hits[from_: from_ + size]

    # If intent-detected filters produced zero results, attempt incremental relaxation
    if total == 0 and intent.get("filters") and use_intent_parsing:
        logger.info("No hits with strict intent filters; attempting incremental relaxation.")

        def _run_with_intent(local_intent: dict) -> tuple[list, int, list]:
            """Run the chosen search strategy given a particular intent object.

            Returns (hits, total, filters_used)
            """
            # build filters for this local intent
            local_filters = []
            if local_intent["filters"].get("procurement_method"):
                local_filters.append({"term": {"procurement_method": local_intent["filters"]["procurement_method"]}})
            if local_intent["filters"].get("contract_status"):
                local_filters.append({"term": {"contract_status": local_intent["filters"]["contract_status"]}})
            if local_intent["filters"].get("buyer_name"):
                local_filters.append({"match": {"buyer_name.text": local_intent["filters"]["buyer_name"]}})
            if local_intent["filters"].get("is_framework"):
                local_filters.append({"term": {"is_framework": True}})
            if min_value is not None or max_value is not None:
                range_q = {}
                if min_value is not None:
                    range_q["gte"] = min_value
                if max_value is not None:
                    range_q["lte"] = max_value
                local_filters.append({"range": {"value_amount": range_q}})

            local_location_boosts = [
                {
                    "multi_match": {
                        "query": loc,
                        "fields": ["chunk_text", "description", "buyer_name.text"],
                        "boost": 1.5,
                    }
                }
                for loc in local_intent.get("locations", [])
            ]

            local_embed_text = local_intent.get("clean_text") or query_text or ""

            local_service_boosts = []
            for svc in local_intent.get("services", []) or []:
                local_service_boosts.append({
                    "match_phrase": {"chunk_text": {"query": svc, "boost": 2.0}}
                })
                local_service_boosts.append({
                    "match_phrase": {"title": {"query": svc, "boost": 2.0}}
                })

            # If empty query text, use browse
            if not (local_embed_text or "").strip():
                hits_local, total_local = _browse_search(es, local_filters, page, size, sort=sort)
                return hits_local, total_local, local_filters

            if search_type == "text":
                hits_local = _text_search(es, local_embed_text, local_filters, local_location_boosts, size, service_boosts=local_service_boosts, sort=sort)
                total_local = len(hits_local)
            elif search_type == "semantic":
                qv = embedder.encode(local_embed_text, normalize_embeddings=True).tolist()
                hits_local = _semantic_search(es, qv, local_filters, size)
                if sort and sort != "relevance":
                    hits_local = _sort_hits(hits_local, sort)
                total_local = len(hits_local)
            else:
                qv = embedder.encode(local_embed_text, normalize_embeddings=True).tolist()
                fetch_size = max(size * 3, 50)
                bm25_hits_local = _text_search(es, local_embed_text, local_filters, local_location_boosts, fetch_size, service_boosts=local_service_boosts, sort=sort)
                knn_hits_local = _semantic_search(es, qv, local_filters, fetch_size)
                fused_local = reciprocal_rank_fusion([bm25_hits_local, knn_hits_local], k=rrf_k)
                if sort and sort != "relevance":
                    fused_local = _sort_hits(fused_local, sort)
                hits_local = fused_local[:size]
                total_local = len(fused_local)

            return hits_local, total_local, local_filters

        # Try relaxing filters in order of specificity
        relax_order = ["buyer_name", "contract_status", "procurement_method", "is_framework"]
        for key in relax_order:
            if key in intent["filters"]:
                # make a shallow copy and remove this key
                new_intent = {k: v for k, v in intent.items()}
                new_filters = dict(intent["filters"]) if intent.get("filters") else {}
                new_filters.pop(key, None)
                new_intent["filters"] = new_filters
                hits_try, total_try, filters_used = _run_with_intent(new_intent)
                if total_try > 0:
                    intent["fallback"] = f"relaxed_{key}"
                    hits = hits_try
                    total = total_try
                    aggs = _run_aggregations(es, filters_used)
                    break

        # If still no results after relaxations, fall back to disabling intent parsing entirely
        if total == 0:
            logger.info("Relaxations failed; retrying without intent parsing.")
            intent["fallback"] = "retrial_no_filters"
            return search_contracts(
                es=es,
                query_text=query_text,
                procurement_method=None,
                contract_status=None,
                min_value=min_value,
                max_value=max_value,
                search_type=search_type,
                sort=sort,
                page=page,
                size=size,
                use_intent_parsing=False,
                rrf_k=rrf_k,
            )

    return _format_response(hits, total, aggs, intent)


# ---------------------------------------------------------------------------
# Internal search helpers
# ---------------------------------------------------------------------------

def _text_search(
    es: Elasticsearch,
    query_text: str,
    filters: list,
    location_boosts: list,
    size: int,
    service_boosts: list | None = None,
    sort: str = "relevance",
) -> list[dict]:
    should_clauses = []
    if location_boosts:
        should_clauses.extend(location_boosts)
    if service_boosts:
        should_clauses.extend(service_boosts)

    body = {
        "size": size,
        "query": {
            "bool": {
                "must": {
                    "multi_match": {
                        "query": query_text,
                        "fields": [
                            "title^4",
                            "buyer_name.text^3",
                            "cpv_descriptions^2",
                            "cpv_codes^2",
                            "classification_codes^2",
                            "description^2",
                            "chunk_text",
                        ],
                        "type": "best_fields",
                        "fuzziness": "AUTO",
                        "minimum_should_match": "60%",
                    }
                },
                "should": should_clauses,
                "filter": filters,
            }
        },
    }
    # apply server-side sort when requested (non-relevance)
    if sort and sort != "relevance":
        sort_clause = None
        if sort == "date_desc":
            sort_clause = [{"release_date": {"order": "desc", "missing": "_last"}}]
        elif sort == "date_asc":
            sort_clause = [{"release_date": {"order": "asc", "missing": "_last"}}]
        elif sort == "value_desc":
            sort_clause = [{"value_amount": {"order": "desc", "missing": "_last"}}]
        elif sort == "value_asc":
            sort_clause = [{"value_amount": {"order": "asc", "missing": "_last"}}]
        elif sort == "buyer_asc":
            sort_clause = [{"buyer_name": {"order": "asc", "missing": "_last"}}]
        elif sort == "buyer_desc":
            sort_clause = [{"buyer_name": {"order": "desc", "missing": "_last"}}]

        if sort_clause is not None:
            body["sort"] = sort_clause

    resp = es.search(index=INDEX_NAME, body=body)
    return resp["hits"]["hits"]


def _sort_hits(hits: list[dict], sort: str) -> list[dict]:
    """Sort a list of ES hits (in-memory) according to the requested sort key."""
    if not hits:
        return hits

    reverse = False
    key = None

    if sort == "date_desc":
        reverse = True
        key = lambda h: (h.get("_source", {}).get("release_date") or "")
    elif sort == "date_asc":
        key = lambda h: (h.get("_source", {}).get("release_date") or "")
    elif sort == "value_desc":
        reverse = True
        key = lambda h: (h.get("_source", {}).get("value_amount") or 0)
    elif sort == "value_asc":
        key = lambda h: (h.get("_source", {}).get("value_amount") or 0)
    elif sort == "buyer_asc":
        key = lambda h: (h.get("_source", {}).get("buyer_name") or "")
    elif sort == "buyer_desc":
        reverse = True
        key = lambda h: (h.get("_source", {}).get("buyer_name") or "")
    else:
        return hits

    try:
        return sorted(hits, key=key, reverse=reverse)
    except Exception:
        return hits


def _semantic_search(
    es: Elasticsearch,
    query_vector: list,
    filters: list,
    size: int,
) -> list[dict]:
    """
    Runs kNN search with filters applied INSIDE the knn block (pre-filter).
    This ensures k documents are always returned, unlike post_filter which
    would shrink the result set after the fact.
    """
    knn = {
        "field": "embedding",
        "query_vector": query_vector,
        "k": size,
        "num_candidates": max(size * 50, 500),
    }
    if filters:
        knn["filter"] = {"bool": {"filter": filters}}

    body = {"size": size, "knn": knn}
    resp = es.search(index=INDEX_NAME, body=body)
    return resp["hits"]["hits"]


def _browse_search(
    es: Elasticsearch,
    filters: list,
    page: int,
    size: int,
    sort: str = "relevance",
) -> tuple[list[dict], int]:
    """
    Returns a plain filtered listing for empty-query browsing with accurate totals.
    """
    from_ = (page - 1) * size
    body = {
        "from": from_,
        "size": size,
        "track_total_hits": True,
        "query": {"bool": {"filter": filters}} if filters else {"match_all": {}},
    }

    if sort and sort != "relevance":
        if sort == "date_desc":
            body["sort"] = [{"release_date": {"order": "desc", "missing": "_last"}}]
        elif sort == "date_asc":
            body["sort"] = [{"release_date": {"order": "asc", "missing": "_last"}}]
        elif sort == "value_desc":
            body["sort"] = [{"value_amount": {"order": "desc", "missing": "_last"}}]
        elif sort == "value_asc":
            body["sort"] = [{"value_amount": {"order": "asc", "missing": "_last"}}]
        elif sort == "buyer_asc":
            body["sort"] = [{"buyer_name": {"order": "asc", "missing": "_last"}}]
        elif sort == "buyer_desc":
            body["sort"] = [{"buyer_name": {"order": "desc", "missing": "_last"}}]
    else:
        body["sort"] = [
            {"release_date": {"order": "desc", "missing": "_last"}},
            {"ocid": {"order": "asc"}},
        ]
    resp = es.search(index=INDEX_NAME, body=body)
    hits = resp.get("hits", {}).get("hits", [])
    total_obj = resp.get("hits", {}).get("total", 0)
    total = total_obj.get("value", 0) if isinstance(total_obj, dict) else int(total_obj or 0)
    return hits, total


def _run_aggregations(es: Elasticsearch, filters: list) -> dict:
    """
    Runs a separate lightweight query to compute facets.
    Kept separate so RRF pagination doesn't interfere with agg counts.
    """
    body = {
        "size": 0,
        "query": {"bool": {"filter": filters}} if filters else {"match_all": {}},
        "aggs": {
            "procurement_methods": {"terms": {"field": "procurement_method", "size": 20}},
            "contract_statuses":   {"terms": {"field": "contract_status",   "size": 10}},
            "top_buyers":          {"terms": {"field": "buyer_name",        "size": 10}},
            "value_stats":         {"stats": {"field": "value_amount"}},
            "cpv_codes":           {"terms": {"field": "cpv_codes", "size": 20}},
        },
    }
    resp = es.search(index=INDEX_NAME, body=body)
    return resp.get("aggregations", {})

def _normalize_score(hit: dict) -> float:
    """
    Normalize relevance scores to 0-1 range for consistent comparison across search types.
    
    - RRF scores: 1/(k+rank), typical range 0.016-0.033 → scale to 0-1
    - BM25 scores: 0-50+ → cap at 20 and normalize  
    - Cosine similarity: 0-1 → already normalized
    """
    if hit.get("_rrf_score") is not None:
        # RRF score: scale from ~0.03 max to 1.0
        # Max possible RRF = 1/61 + 1/61 ≈ 0.0328 (top rank in both lists)
        rrf = hit["_rrf_score"]
        normalized = min(rrf / 0.033, 1.0)  # Cap at 1.0
        return round(normalized, 4)
    
    if hit.get("_score") is not None:
        es_score = hit["_score"]
        # For semantic search, _score is already cosine similarity (0-1)
        # For BM25, _score can be very high, so cap and normalize
        if es_score <= 1.0:
            # Already normalized (semantic search)
            return round(es_score, 4)
        else:
            # BM25 score: normalize to 0-1 with reasonable cap
            # Most BM25 scores are 0-20, but can exceed 50
            normalized = min(es_score / 20.0, 1.0)
            return round(normalized, 4)
    
    return 0.0


# ---------------------------------------------------------------------------
# Response formatting
# ---------------------------------------------------------------------------

def _format_response(hits: list, total: int, aggs: dict, intent: dict) -> dict:
    results = []
    for rank, hit in enumerate(hits, start=1):
        source = hit.get("_source", {})
        # Build a summary from description, chunk_text, or title; prefer longer content
        summary = (
            (source.get("description") or "")[:400] or
            (source.get("chunk_text") or "")[:300] or
            source.get("title") or
            "No summary available"
        )
        results.append({
            "contract_id":        source.get("contract_id"),
            "notice_id":          source.get("notice_id"),
            "ocid":               source.get("ocid"),
            "title":              source.get("title"),
            "summary":            summary,
            "buyer_name":         source.get("buyer_name"),
            "supplier_names":     source.get("supplier_names", []),
            "contract_status":    source.get("contract_status"),
            "value_amount":       source.get("value_amount"),
            "value_currency":     source.get("value_currency"),
            "release_date":       source.get("release_date"),
            "procurement_method": source.get("procurement_method"),
            "cpv_codes":          source.get("cpv_codes", []),
            "cpv_descriptions":   source.get("cpv_descriptions", []),
            "classification_codes": source.get("classification_codes", []),
            "relevance_score":    _normalize_score(hit),
            "explanation":        _build_explanation(hit, rank, intent),
        })

    facets = {
        "procurement_methods": _extract_buckets(aggs, "procurement_methods"),
        "contract_statuses":   _extract_buckets(aggs, "contract_statuses"),
        "top_buyers":          _extract_buckets(aggs, "top_buyers"),
        "value_stats":         aggs.get("value_stats", {}),
    }

    return {
        "total":         total,
        "results":       results,
        "facets":        facets,
        "parsed_intent": intent,
    }


def _build_explanation(hit: dict, rank: int, intent: dict) -> dict:
    source = hit.get("_source", {})
    
    # --- Metadata filters matched ---
    metadata_filters = []
    if intent.get("filters", {}).get("procurement_method"):
        v = intent["filters"]["procurement_method"]
        if source.get("procurement_method") == v:
            metadata_filters.append(f"Procurement method: {v}")
    if intent.get("filters", {}).get("buyer_name"):
        v = intent["filters"]["buyer_name"]
        if source.get("buyer_name") == v:
            metadata_filters.append(f"Buyer: {v}")
    if intent.get("filters", {}).get("contract_status"):
        v = intent["filters"]["contract_status"]
        if source.get("contract_status") == v:
            metadata_filters.append(f"Status: {v}")
    if intent.get("filters", {}).get("is_framework"):
        if source.get("is_framework"):
            metadata_filters.append("Is framework agreement")

    # --- Keyword overlap in content ---
    query_words = set(
        w for w in (intent.get("clean_text") or intent.get("raw") or "").lower().split()
        if len(w) > 3
    )
    content = (source.get("chunk_text") or source.get("description") or "").lower()
    keyword_overlap = [w for w in query_words if w in content]

    # --- Location mentions in content ---
    location_mentions = []
    for loc in intent.get("locations", []):
        if loc.lower() in content:
            location_mentions.append(loc)

    # --- Semantic signal ---
    semantic_signal = {
        "method": "embedding cosine similarity via HNSW kNN",
        "combined_with": "BM25 text relevance via hybrid RRF ranking"
    }

    return {
        "rank": rank,
        "scores": {
            "rrf_score": round(hit.get("_rrf_score"), 4) if hit.get("_rrf_score") else None,
            "bm25_score": round(hit.get("_score"), 4) if hit.get("_score") else None,
        },
        "why_it_matched": {
            "metadata_filters": metadata_filters if metadata_filters else ["No explicit filter matches"],
            "keyword_overlap": keyword_overlap if keyword_overlap else ["No keyword overlap in content"],
            "location_mentions": location_mentions if location_mentions else [],
            "semantic_similarity": semantic_signal,
        },
    }


def _extract_buckets(aggs: dict, key: str) -> list:
    return [
        {"value": b["key"], "count": b["doc_count"]}
        for b in aggs.get(key, {}).get("buckets", [])
    ]


if __name__ == "__main__":
    # Quick smoke checks for parse_query_intent
    examples = [
        "Multi-supplier frameworks for cloud services in London",
        "NHS open tenders for IT services",
        "awarded framework agreement for construction in Manchester",
    ]
    for q in examples:
        print("Query:", q)
        intent = parse_query_intent(q)
        print("Parsed intent:", intent)
        print("---")
