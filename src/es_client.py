import json
import logging
from elasticsearch import Elasticsearch, helpers

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

INDEX_NAME = "fts_contracts"
EMBEDDING_DIM = 384  # Dimension for all-MiniLM-L6-v2

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
    actions = [
        {
            "_index": INDEX_NAME,
            "_source": doc
        }
        for doc in documents
    ]
    
    success, failed = helpers.bulk(es, actions, stats_only=True)
    logger.info(f"Indexed {success} documents successfully. Failed: {failed}")
    return success
