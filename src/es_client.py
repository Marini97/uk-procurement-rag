import logging
import spacy
from typing import Optional
from elasticsearch import Elasticsearch, helpers
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

INDEX_NAME = "fts_contracts"
EMBEDDING_DIM = 384  # all-MiniLM-L6-v2

embedder = SentenceTransformer("all-MiniLM-L6-v2")

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    logger.warning("spaCy model not found. Run: python -m spacy download en_core_web_sm")
    nlp = None

PROCUREMENT_METHOD_KEYWORDS = {
    "framework agreement": "Framework Agreement",
    "framework": "Framework Agreement",
    "open procedure": "open",
    "open tender": "open",
    "direct award": "direct",
    "direct": "direct",
    "restricted": "selective",
    "negotiated": "negotiated",
    "competitive dialogue": "competitive dialogue",
    "dynamic purchasing": "dynamic purchasing system",
}

SERVICE_KEYWORDS = {
    # common natural-language service/categories -> search keywords
    "it services": "IT services",
    "information technology": "IT services",
    "cloud services": "cloud services",
    "cloud": "cloud services",
    "construction": "construction",
    "architectural": "construction",
    "facilities management": "facilities management",
    "consultancy": "consultancy",
    "legal": "legal services",
}

BUYER_KEYWORDS = {
    "nhs": "NHS",
    "national health service": "NHS",
    "nhs england": "NHS England",
}

CONTRACT_STATUS_KEYWORDS = {
    "awarded": "active",
    "award": "active",
    "awards": "active",
}


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

    success, failed = helpers.bulk(
        es,
        actions,
        stats_only=True,
        raise_on_error=False,
        chunk_size=500,
        request_timeout=120,
    )
    if failed:
        logger.warning(f"Bulk index: {success} OK, {failed} FAILED")
    else:
        logger.info(f"Bulk index: {success} documents indexed successfully")
    return success


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
    intent = {
        "raw": query_text,
        "filters": {},
        "locations": [],
        "clean_text": query_text,
    }

    lower = query_text.lower()

    # Match longest keyword first to avoid partial matches
    for kw in sorted(PROCUREMENT_METHOD_KEYWORDS, key=len, reverse=True):
        if kw in lower:
            intent["filters"]["procurement_method"] = PROCUREMENT_METHOD_KEYWORDS[kw]
            intent["clean_text"] = intent["clean_text"].replace(kw, "").strip()
            break

    if nlp is not None:
        doc = nlp(query_text)
        for ent in doc.ents:
            if ent.label_ == "ORG" and "buyer_name" not in intent["filters"]:
                intent["filters"]["buyer_name"] = ent.text
            elif ent.label_ in ("GPE", "LOC"):
                intent["locations"].append(ent.text)

    # Detect simple buyer keywords (e.g., "NHS") and normalise
    for kw, canonical in BUYER_KEYWORDS.items():
        if kw in lower:
            intent["filters"]["buyer_name"] = canonical
            intent["clean_text"] = intent["clean_text"].replace(kw, "")
            break

    # Detect services/categories mentioned in the query and remove them from clean_text
    services = []
    for kw, label in SERVICE_KEYWORDS.items():
        if kw in lower:
            services.append(label)
            intent["clean_text"] = intent["clean_text"].replace(kw, "")
    if services:
        intent["services"] = services

    # Detect contract-status words like 'awarded' -> map to contract_status filter
    for kw, status in CONTRACT_STATUS_KEYWORDS.items():
        if kw in lower:
            intent["filters"]["contract_status"] = status
            intent["clean_text"] = intent["clean_text"].replace(kw, "")
            break

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
        filters.append({"term": {"buyer_name": intent["filters"]["buyer_name"]}})
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
                "fields": ["chunk_text", "description", "buyer_name.text"],
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
        hits, total = _browse_search(es, filters, page, size)
        aggs = _run_aggregations(es, filters)
        return _format_response(hits, total, aggs, intent)

    # --- Dispatch to search strategy ---
    if search_type == "text":
        hits = _text_search(es, embed_text, filters, location_boosts, size, service_boosts=service_boosts)
        total = len(hits)

    elif search_type == "semantic":
        query_vector = embedder.encode(embed_text, normalize_embeddings=True).tolist()
        hits = _semantic_search(es, query_vector, filters, size)
        total = len(hits)

    else:  # hybrid — two queries + client-side RRF
        query_vector = embedder.encode(embed_text, normalize_embeddings=True).tolist()

        # Fetch more candidates per leg so RRF has enough to work with
        fetch_size = max(size * 3, 50)

        bm25_hits = _text_search(es, embed_text, filters, location_boosts, fetch_size, service_boosts=service_boosts)
        knn_hits  = _semantic_search(es, query_vector, filters, fetch_size)

        fused = reciprocal_rank_fusion([bm25_hits, knn_hits], k=rrf_k)
        hits  = fused[:size]
        total = len(fused)

    # Aggregations (separate lightweight query)
    aggs = _run_aggregations(es, filters)

    # Paginate for non-hybrid (hybrid is already sliced above)
    if search_type != "hybrid":
        from_ = (page - 1) * size
        hits = hits[from_: from_ + size]

    # If intent-detected filters produced zero results, retry with relaxed parsing
    if total == 0 and intent.get("filters") and use_intent_parsing:
        logger.info("No hits with strict intent filters; retrying without intent parsing.")
        # Mark that we're falling back so callers/UI can surface it
        intent["fallback"] = "retrial_no_filters"
        return search_contracts(
            es=es,
            query_text=query_text,
            procurement_method=None,
            contract_status=None,
            min_value=min_value,
            max_value=max_value,
            search_type=search_type,
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
    resp = es.search(index=INDEX_NAME, body=body)
    return resp["hits"]["hits"]


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
        "num_candidates": max(size * 10, 100),
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
        "sort": [
            {"release_date": {"order": "desc", "missing": "_last"}},
            {"ocid": {"order": "asc"}},
        ],
    }
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
        },
    }
    resp = es.search(index=INDEX_NAME, body=body)
    return resp.get("aggregations", {})


# ---------------------------------------------------------------------------
# Response formatting
# ---------------------------------------------------------------------------

def _format_response(hits: list, total: int, aggs: dict, intent: dict) -> dict:
    results = []
    for rank, hit in enumerate(hits, start=1):
        source = hit.get("_source", {})
        results.append({
            "notice_id":          source.get("notice_id"),
            "ocid":               source.get("ocid"),
            "title":              source.get("title"),
            "buyer_name":         source.get("buyer_name"),
            "supplier_names":     source.get("supplier_names", []),
            "contract_status":    source.get("contract_status"),
            "value_amount":       source.get("value_amount"),
            "value_currency":     source.get("value_currency"),
            "release_date":       source.get("release_date"),
            "procurement_method": source.get("procurement_method"),
            "cpv_codes":          source.get("cpv_codes", []),
            "cpv_descriptions":   source.get("cpv_descriptions", []),
            "chunk_text":         (source.get("chunk_text") or "")[:300],
            "relevance_score":    hit.get("_rrf_score") or hit.get("_score"),
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
    filters_matched = []

    if intent.get("filters", {}).get("procurement_method"):
        v = intent["filters"]["procurement_method"]
        if source.get("procurement_method") == v:
            filters_matched.append(f"procurement_method = {v}")

    if intent.get("filters", {}).get("buyer_name"):
        v = intent["filters"]["buyer_name"]
        if source.get("buyer_name") == v:
            filters_matched.append(f"buyer_name = {v}")

    for loc in intent.get("locations", []):
        if loc.lower() in (source.get("chunk_text") or "").lower():
            filters_matched.append(f"location mention: {loc}")

    query_words = set(
        w for w in (intent.get("clean_text") or intent.get("raw") or "").lower().split()
        if len(w) > 3
    )
    keyword_overlap = [w for w in query_words if w in (source.get("chunk_text") or "").lower()]

    return {
        "rank":                      rank,
        "rrf_score":                 hit.get("_rrf_score"),
        "bm25_score":                hit.get("_score"),
        "metadata_filters_matched":  filters_matched,
        "keyword_overlap":           keyword_overlap,
        "semantic_signal":           "embedding cosine similarity via HNSW kNN",
    }


def _extract_buckets(aggs: dict, key: str) -> list:
    return [
        {"value": b["key"], "count": b["doc_count"]}
        for b in aggs.get(key, {}).get("buckets", [])
    ]
