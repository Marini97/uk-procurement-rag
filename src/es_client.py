import json
import logging
from elasticsearch import Elasticsearch, helpers
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

INDEX_NAME = "fts_contracts"
EMBEDDING_DIM = 384  # Dimension for all-MiniLM-L6-v2

# Instantiate the sentence transformer model
embedder = SentenceTransformer('all-MiniLM-L6-v2')

def get_es_client():
    return Elasticsearch(
        "http://localhost:9200",
        request_timeout=30,
        max_retries=3,
        retry_on_timeout=True
    )

def setup_index(es: Elasticsearch):
    """Creates the Elasticsearch index with appropriate mappings for Hybrid Search."""
    if es.indices.exists(index=INDEX_NAME):
        logger.info(f"Index {INDEX_NAME} already exists. Deleting it for a fresh start.")
        es.indices.delete(index=INDEX_NAME)

    # Define the mapping for dense spaces and keyword/texts
    mapping = {
        "mappings": {
            "properties": {
                "ocid": {"type": "keyword"},
                "notice_id": {"type": "keyword"},
                "release_date": {"type": "date"},
                
                "tender_id": {"type": "keyword"},
                "title": {"type": "text"},
                "description": {"type": "text"},
                "procurement_method": {"type": "keyword"},
                
                "buyer_name": {"type": "keyword"},
                "supplier_names": {"type": "keyword"},
                
                "contract_id": {"type": "keyword"},
                "contract_status": {"type": "keyword"},
                "value_amount": {"type": "float"},
                "value_currency": {"type": "keyword"},
                
                "cpv_codes": {"type": "keyword"},
                "cpv_descriptions": {"type": "text"},
                
                "chunk_text": {"type": "text"},  # For BM25 scoring
                "embedding": {
                    "type": "dense_vector",
                    "dims": EMBEDDING_DIM,
                    "index": True,
                    "similarity": "cosine"
                }
            }
        }
    }

    es.indices.create(index=INDEX_NAME, body=mapping)
    logger.info(f"Successfully created index {INDEX_NAME}.")

def bulk_index_docs(es: Elasticsearch, documents: list):
    """Bulk indexes a list of normalized contract documents."""
    actions = []
    for doc in documents:
        action = {
            "_index": INDEX_NAME,
            "_source": doc
        }
        if "_id" in doc:
            action["_id"] = doc.pop("_id") # Remove _id from source and set as ES meta _id
        actions.append(action)
    
    success, failed = helpers.bulk(es, actions, stats_only=True)
    logger.info(f"Indexed {success} documents successfully. Failed: {failed}")
    return success

def search_contracts(
    es: Elasticsearch,
    query_text: str = None,
    procurement_method: str = None,
    contract_status: str = None,
    search_type: str = "hybrid",
    page: int = 1,
    size: int = 10
):
    """
    Search contracts with support for text (BM25), semantic (k-NN), or hybrid search.
    Includes filtering, aggregations, and pagination.
    """
    from_ = (page - 1) * size
    
    # Build filters
    filters = []
    if procurement_method:
        filters.append({"term": {"procurement_method": procurement_method}})
    if contract_status:
        filters.append({"term": {"contract_status": contract_status}})
        
    query_body = {"bool": {}}
    
    knn_body = None
    if query_text:
        if search_type in ["text", "hybrid"]:
            query_body["bool"]["must"] = {
                "multi_match": {
                    "query": query_text,
                    "fields": ["title", "description", "chunk_text", "cpv_descriptions"]
                }
            }
        
        if search_type in ["semantic", "hybrid"]:
            # Generate embedding for the query
            query_vector = embedder.encode(query_text).tolist()
            knn_body = {
                "field": "embedding",
                "query_vector": query_vector,
                "k": size,
                "num_candidates": 100,
                "filter": filters if filters else []
            }
    else:
        query_body["bool"]["must"] = {"match_all": {}}
        
    body = {
        "from": from_,
        "size": size,
        "query": query_body,
        "aggs": {
            "procurement_methods": {"terms": {"field": "procurement_method"}},
            "contract_statuses": {"terms": {"field": "contract_status"}}
        }
    }

    # Apply filters as post_filter so aggregations aren't scoped down by UI selections
    if filters:
        body["post_filter"] = {"bool": {"filter": filters}}
    
    if knn_body:
        body["knn"] = knn_body
        
    response = es.search(index=INDEX_NAME, body=body)
    return response
