import json
import logging
import sys
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from es_client import get_es_client, setup_index, bulk_index_docs, restore_index_settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Small and fast model for generating local embeddings
MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 2000

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
    tender = release.get("tender") or {}
    tender_id = tender.get("id") or ""
    title = tender.get("title") or "Untitled"
    description = tender.get("description") or ""
    proc_method = tender.get("procurementMethod") or "Unknown"
    
    # Categories / CPV
    items = tender.get("items", [])
    cpv_codes = []
    cpv_descriptions = []
    for item in items:
        clazz = item.get("classification", {})
        if clazz.get("id"): cpv_codes.append(clazz["id"])
        if clazz.get("description"): cpv_descriptions.append(clazz["description"])
        
    # Parties (Buyers / Suppliers)
    parties = release.get("parties") or []
    party_map = {p.get("id"): p for p in parties if p.get("id")}
    buyers = [p for p in parties if "buyer" in (p.get("roles") or [])]
    buyer_party = buyers[0] if buyers else {}
    buyer_name = buyer_party.get("name") if buyer_party else "Unknown"
    buyer_id = buyer_party.get("id") if buyer_party else None
    # buyer contact/address
    contact = buyer_party.get("contactPoint") if buyer_party else {}
    buyer_contact = ", ".join(str(contact.get(k)) for k in ("name", "email", "telephone") if contact and contact.get(k)) if contact else ""
    address = buyer_party.get("address") if buyer_party else {}
    if address:
        # Try common address fields
        buyer_address = ", ".join(str(address.get(k)) for k in ("streetAddress", "locality", "region", "countryName") if address.get(k))
    else:
        buyer_address = ""
    
    # Awards & Suppliers mapped by award ID
    awards = release.get("awards") or []
    award_map = {}
    for award in awards:
        suppliers = award.get("suppliers") or []
        supplier_names = [s.get("name") for s in suppliers if s.get("name")]
        supplier_ids = [s.get("id") for s in suppliers if s.get("id")]
        award_map[award.get("id")] = {
            "supplier_names": supplier_names,
            "supplier_ids": supplier_ids,
            "award_date": award.get("date"),
            "award_value": (award.get("value") or {}).get("amount"),
            "award_value_currency": (award.get("value") or {}).get("currency"),
        }

    contracts = release.get("contracts") or []
    
    docs = []
    
    if not contracts:
        # If no contracts yet (e.g. tender phase), generate a doc for the tender
        docs.append(create_doc(
            ocid=ocid,
            notice_id=notice_id,
            release_date=release_date,
            tender_id=tender_id,
            title=title,
            description=description,
            proc_method=proc_method,
            buyer_name=buyer_name,
            buyer_id=buyer_id,
            buyer_contact=buyer_contact,
            buyer_address=buyer_address,
            supplier_names=[],
            supplier_ids=[],
            supplier_countries=[],
            supplier_count=0,
            contract_id="",
            contract_status="",
            value_amount=0.0,
            value_currency="",
            cpv_codes=cpv_codes,
            cpv_descriptions=cpv_descriptions,
            award_date=None,
            award_value_amount=None,
            award_value_currency=None,
            tender_period_start=(tender.get("tenderPeriod") or {}).get("startDate"),
            tender_period_end=(tender.get("tenderPeriod") or {}).get("endDate"),
            lots=tender.get("lots") or [],
            items_text=", ".join([it.get("description") or "" for it in items]) if items else "",
            classification_codes=cpv_codes,
            procurement_category=tender.get("mainProcurementCategory"),
            is_framework=("framework" in (proc_method or "").lower() or "framework" in (tender.get("procurementMethodDetails") or "").lower()),
            url=(release.get("documents") or [])[0].get("url") if release.get("documents") else None,
            documents_text="; ".join([d.get("title","") for d in (release.get("documents") or [])]),
            notes=", ".join(release.get("tag") or []) if release.get("tag") else "",
        ))
    else:
        # Generate a doc per contract
        for contract in contracts:
            contract_id = contract.get("id") or ""
            contract_status = contract.get("status") or "Unknown"
            
            val = contract.get("value") or {}
            value_amount = val.get("amount") or 0.0
            value_currency = val.get("currency") or ""
            
            # Link to award suppliers
            award_id = contract.get("awardID", "")
            award_info = award_map.get(award_id, {})
            supplier_names = award_info.get("supplier_names") or []
            supplier_ids = award_info.get("supplier_ids") or []
            award_date = award_info.get("award_date")
            award_value_amount = award_info.get("award_value")
            award_value_currency = award_info.get("award_value_currency")

            # Supplier countries from party_map where available
            supplier_countries = []
            for sid in supplier_ids:
                p = party_map.get(sid) or {}
                addr = p.get("address") or {}
                country = addr.get("countryName") or addr.get("addressDetails", {}).get("country") if addr else None
                if country:
                    supplier_countries.append(country)

            supplier_count = len(set(supplier_ids)) if supplier_ids else (len(supplier_names) if supplier_names else 0)
            
            docs.append(create_doc(
                ocid=ocid,
                notice_id=notice_id,
                release_date=release_date,
                tender_id=tender_id,
                title=title,
                description=description,
                proc_method=proc_method,
                buyer_name=buyer_name,
                buyer_id=buyer_id,
                buyer_contact=buyer_contact,
                buyer_address=buyer_address,
                supplier_names=supplier_names,
                supplier_ids=supplier_ids,
                supplier_countries=supplier_countries,
                supplier_count=supplier_count,
                contract_id=contract_id,
                contract_status=contract_status,
                value_amount=value_amount,
                value_currency=value_currency,
                cpv_codes=cpv_codes,
                cpv_descriptions=cpv_descriptions,
                award_date=award_date,
                award_value_amount=award_value_amount,
                award_value_currency=award_value_currency,
                tender_period_start=(tender.get("tenderPeriod") or {}).get("startDate"),
                tender_period_end=(tender.get("tenderPeriod") or {}).get("endDate"),
                lots=tender.get("lots") or [],
                items_text=", ".join([it.get("description") or "" for it in items]) if items else "",
                classification_codes=cpv_codes,
                procurement_category=tender.get("mainProcurementCategory"),
                is_framework=("framework" in (proc_method or "").lower() or "framework" in (tender.get("procurementMethodDetails") or "").lower()),
                url=(release.get("documents") or [])[0].get("url") if release.get("documents") else None,
                documents_text="; ".join([d.get("title","") for d in (release.get("documents") or [])]),
                notes=", ".join(release.get("tag") or []) if release.get("tag") else "",
            ))

    return docs

def create_doc(
    ocid,
    notice_id,
    release_date,
    tender_id,
    title,
    description,
    proc_method,
    buyer_name,
    buyer_id=None,
    buyer_contact=None,
    buyer_address=None,
    supplier_names=None,
    supplier_ids=None,
    supplier_countries=None,
    supplier_count=0,
    contract_id=None,
    contract_status=None,
    value_amount=None,
    value_currency=None,
    cpv_codes=None,
    cpv_descriptions=None,
    award_date=None,
    award_value_amount=None,
    award_value_currency=None,
    tender_period_start=None,
    tender_period_end=None,
    lots=None,
    items_text=None,
    classification_codes=None,
    procurement_category=None,
    is_framework=False,
    url=None,
    documents_text=None,
    notes=None,
):

    supplier_names = supplier_names or []
    supplier_ids = supplier_ids or []
    supplier_countries = supplier_countries or []
    cpv_descriptions = cpv_descriptions or []
    cpv_codes = cpv_codes or []
    lots = lots or []

    supplier_text = ", ".join(supplier_names) if supplier_names else "Unknown Supplier"
    cpv_text = ", ".join(cpv_descriptions) if cpv_descriptions else "No classification"

    # Synthesize the text representing this chunk for embedding & semantic search
    chunk_text = f"Tender Title: {title}\n"
    chunk_text += f"Buyer: {buyer_name or 'Unknown Buyer'}\n"
    if buyer_id:
        chunk_text += f"Buyer ID: {buyer_id}\n"
    if buyer_contact:
        chunk_text += f"Buyer Contact: {buyer_contact}\n"
    if buyer_address:
        chunk_text += f"Buyer Address: {buyer_address}\n"
    if contract_id:
        chunk_text += f"Contract awarded to: {supplier_text}\n"
        chunk_text += f"Contract Status: {contract_status}\n"
    if supplier_count:
        chunk_text += f"Supplier count: {supplier_count}\n"
    if supplier_countries:
        chunk_text += f"Supplier countries: {', '.join(supplier_countries)}\n"
    chunk_text += f"Procurement Method: {proc_method}\n"
    chunk_text += f"Categories: {cpv_text}\n"
    if award_date:
        chunk_text += f"Award date: {award_date}\n"
    if award_value_amount:
        chunk_text += f"Award value: {award_value_amount} {award_value_currency or ''}\n"
    if tender_period_start or tender_period_end:
        chunk_text += f"Tender period: {tender_period_start or ''} - {tender_period_end or ''}\n"
    if lots:
        lot_texts = []
        for lot in lots:
            lot_texts.append(f"{lot.get('id') or ''}: {lot.get('title') or ''} ({(lot.get('value') or {}).get('amount') or ''} {(lot.get('value') or {}).get('currency') or ''})")
        chunk_text += "Lots: " + "; ".join(lot_texts) + "\n"
    if items_text:
        chunk_text += f"Items: {items_text}\n"
    if description:
        chunk_text += f"Description: {description[:500]}...\n"
    if notes:
        chunk_text += f"Notes: {notes}\n"
    if url:
        chunk_text += f"URL: {url}\n"

    # Generate unique deterministic ID for deduplicating repeated releases in Elasticsearch
    doc_id = f"{ocid}-{notice_id}-{contract_id or 'tender'}"

    return {
        "_id": doc_id,
        "ocid": ocid,
        "notice_id": notice_id,
        "release_date": release_date,
        "tender_id": tender_id,
        "title": title,
        "description": description,
        "procurement_method": proc_method,
        "buyer_name": buyer_name,
        "buyer_id": buyer_id,
        "buyer_contact": buyer_contact,
        "buyer_address": buyer_address,
        "supplier_names": supplier_names,
        "supplier_ids": supplier_ids,
        "supplier_names_text": ", ".join(supplier_names) if supplier_names else None,
        "supplier_countries": supplier_countries,
        "supplier_count": supplier_count,
        "contract_id": contract_id,
        "contract_status": contract_status,
        "value_amount": value_amount,
        "value_currency": value_currency,
        "award_date": award_date,
        "award_value_amount": award_value_amount,
        "award_value_currency": award_value_currency,
        "tender_period_start": tender_period_start,
        "tender_period_end": tender_period_end,
        "lots": lots,
        "items": items_text,
        "cpv_codes": cpv_codes,
        "cpv_descriptions": cpv_descriptions,
        "classification_codes": classification_codes,
        "procurement_category": procurement_category,
        "is_framework": is_framework,
        "framework_duration_months": None,
        "url": url,
        "documents": documents_text,
        "notes": notes,
        "chunk_text": chunk_text,
    }

def main(filepath="data/ocds.jsonl"):
    logger.info("Initializing embedding model...")
    model = SentenceTransformer(MODEL_NAME)
    
    logger.info("Connecting to Elasticsearch...")
    es = get_es_client()
    setup_index(es, recreate=True)
    
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
    # Make newly indexed documents visible and restore index settings
    try:
        es.indices.refresh(index=es.indices.get_alias('*').keys())
    except Exception:
        # Fallback: refresh the target index only
        try:
            es.indices.refresh(index='fts_contracts')
        except Exception:
            pass
    try:
        restore_index_settings(es)
    except Exception:
        logger.warning("Failed to restore index settings automatically; you can run restore_index_settings(es) manually.")

if __name__ == "__main__":
    main()
