import json
import logging
import sys
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from es_client import get_es_client, setup_index, bulk_index_docs

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Small and fast model for generating local embeddings
MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 500

def parse_release(release):
    """
    Parses a single OCDS release and normalizes it to a list of contract-level documents.
    If a tender has multiple contracts, this will return multiple docs. 
    If no contracts exist, it yields a tender-level doc to ensure we still index the opportunity.
    """
    ocid = release.get("ocid", "")
    notice_id = release.get("id", "")
    release_date = release.get("date", None)
    
    # Tender info
    tender = release.get("tender", {})
    tender_id = tender.get("id", "")
    title = tender.get("title", "")
    description = tender.get("description", "")
    proc_method = tender.get("procurementMethod", "")
    
    # Categories / CPV
    items = tender.get("items", [])
    cpv_codes = []
    cpv_descriptions = []
    for item in items:
        clazz = item.get("classification", {})
        if clazz.get("id"): cpv_codes.append(clazz["id"])
        if clazz.get("description"): cpv_descriptions.append(clazz["description"])
        
    # Parties (Buyers)
    parties = release.get("parties", [])
    buyers = [p.get("name") for p in parties if "buyer" in p.get("roles", [])]
    buyer_name = buyers[0] if buyers else ""
    
    # Awards & Suppliers mapped by award ID
    awards = release.get("awards", [])
    award_map = {}
    for award in awards:
        suppliers = award.get("suppliers", [])
        supplier_names = [s.get("name") for s in suppliers if s.get("name")]
        award_map[award.get("id")] = supplier_names

    contracts = release.get("contracts", [])
    
    docs = []
    
    if not contracts:
        # If no contracts yet (e.g. tender phase), generate a doc for the tender
        docs.append(create_doc(
            ocid, notice_id, release_date, tender_id, title, description, 
            proc_method, buyer_name, [], "", "", 0.0, "", cpv_codes, cpv_descriptions
        ))
    else:
        # Generate a doc per contract
        for contract in contracts:
            contract_id = contract.get("id", "")
            contract_status = contract.get("status", "")
            
            val = contract.get("value", {})
            value_amount = val.get("amount", 0.0)
            value_currency = val.get("currency", "")
            
            # Link to award suppliers
            award_id = contract.get("awardID", "")
            supplier_names = award_map.get(award_id, [])
            
            docs.append(create_doc(
                ocid, notice_id, release_date, tender_id, title, description,
                proc_method, buyer_name, supplier_names, contract_id, contract_status,
                value_amount, value_currency, cpv_codes, cpv_descriptions
            ))
            
    return docs

def create_doc(ocid, notice_id, release_date, tender_id, title, description, 
               proc_method, buyer_name, supplier_names, contract_id, contract_status,
               value_amount, value_currency, cpv_codes, cpv_descriptions):
    
    supplier_text = ", ".join(supplier_names) if supplier_names else "Unknown Supplier"
    cpv_text = ", ".join(cpv_descriptions) if cpv_descriptions else "No classification"
    
    # Synthesize the text representing this chunk for embedding & semantic search
    chunk_text = f"Tender Title: {title}\n"
    chunk_text += f"Buyer: {buyer_name or 'Unknown Buyer'}\n"
    if contract_id:
        chunk_text += f"Contract awarded to: {supplier_text}\n"
        chunk_text += f"Contract Status: {contract_status}\n"
    chunk_text += f"Procurement Method: {proc_method}\n"
    chunk_text += f"Categories: {cpv_text}\n"
    if description:
        # Limit description to prevent massive embeddings, first ~500 chars is usually enough for RAG context
        chunk_text += f"Description: {description[:500]}...\n"
        
    return {
        "ocid": ocid,
        "notice_id": notice_id,
        "release_date": release_date,
        "tender_id": tender_id,
        "title": title,
        "description": description,
        "procurement_method": proc_method,
        "buyer_name": buyer_name,
        "supplier_names": supplier_names,
        "contract_id": contract_id,
        "contract_status": contract_status,
        "value_amount": value_amount,
        "value_currency": value_currency,
        "cpv_codes": cpv_codes,
        "cpv_descriptions": cpv_descriptions,
        "chunk_text": chunk_text
    }

def main(filepath="data/ocds.jsonl"):
    logger.info("Initializing embedding model...")
    model = SentenceTransformer(MODEL_NAME)
    
    logger.info("Connecting to Elasticsearch...")
    es = get_es_client()
    setup_index(es)
    
    logger.info(f"Opening data file: {filepath}")
    
    docs_batch = []
    
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(tqdm(f, desc="Processing Lines")):
                try:
                    release = json.loads(line)
                    docs = parse_release(release)
                    docs_batch.extend(docs)
                except json.JSONDecodeError:
                    continue
                
                if len(docs_batch) >= BATCH_SIZE:
                    # Generate embeddings for the batch
                    texts = [doc["chunk_text"] for doc in docs_batch]
                    embeddings = model.encode(texts, show_progress_bar=False)
                    
                    # Attach embeddings
                    for i, doc in enumerate(docs_batch):
                        doc["embedding"] = embeddings[i].tolist()
                        
                    # Bulk index
                    bulk_index_docs(es, docs_batch)
                    docs_batch = []
                    
            # Process remaining docs
            if docs_batch:
                texts = [doc["chunk_text"] for doc in docs_batch]
                embeddings = model.encode(texts, show_progress_bar=False)
                for i, doc in enumerate(docs_batch):
                    doc["embedding"] = embeddings[i].tolist()
                bulk_index_docs(es, docs_batch)
                
    except FileNotFoundError:
        logger.error(f"Could not find the dataset at {filepath}. Did you download it?")
        sys.exit(1)

    logger.info("Ingestion complete!")

if __name__ == "__main__":
    main()
