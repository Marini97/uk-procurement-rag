from fastapi import FastAPI, Request, Query
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from src.es_client import get_es_client, search_contracts, INDEX_NAME
from src.rag import rag_answer
import os
import math

app = FastAPI(title="FTS Contract Search", description="Local search interface for UK public procurement data")

templates = Jinja2Templates(directory="templates")
es = get_es_client()

@app.get("/", response_class=HTMLResponse)
async def index(
    request: Request,
    q: str = Query(None, description="Search query"),
    method: str = Query(None, description="Procurement method filter"),
    status: str = Query(None, description="Contract status filter"),
    search_type: str = Query("hybrid", description="Type of search: text, semantic, or hybrid"),
    rag: str = Query(None, description="Set to 1 to request a RAG answer for the query"),
    page: int = Query(1, ge=1, description="Page number")
):
    size = 10
    
    # Execute search
    try:
        response = search_contracts(
            es=es,
            query_text=q,
            procurement_method=method,
            contract_status=status,
            search_type=search_type,
            page=page,
            size=size
        )
    except Exception as e:
        response = {"total": 0, "results": [], "facets": {}}
        print(f"Error during search: {e}")

    # Primary path: new normalized response shape from es_client.search_contracts
    hits = response.get("results", [])
    total_value = response.get("total", 0)

    # If user requested a RAG answer (and provided a query), run the RAG helper
    rag_response = None
    try:
        if rag and q:
            # pass model via env var RAG_MODEL if set, otherwise fallback to summary
            rag_response = rag_answer(query=q, es=es, top_k=5, model_name=os.environ.get("RAG_MODEL"))
    except Exception as e:
        print(f"RAG generation failed: {e}")

    # Backward-compat path: older raw Elasticsearch response shape
    if not hits and "hits" in response:
        hits = response.get("hits", {}).get("hits", [])
        total_obj = response.get("hits", {}).get("total", {})
        if isinstance(total_obj, dict):
            total_value = total_obj.get("value", 0)
        else:
            total_value = total_obj
        
    # Calculate pages
    total_pages = math.ceil(total_value / size) if total_value > 0 else 0
    total_pages = min(total_pages, 100) # limit to max 100 pages for safety

    # Extract aggregations for filters
    facets = response.get("facets", {})
    methods_agg = [
        {"key": item["value"], "count": item["count"]}
        for item in facets.get("procurement_methods", [])
    ]
    statuses_agg = [
        {"key": item["value"], "count": item["count"]}
        for item in facets.get("contract_statuses", [])
    ]

    # Backward-compat path for legacy aggregation structure
    if not methods_agg and not statuses_agg and "aggregations" in response:
        aggs = response.get("aggregations", {})
        methods_agg = [
            {"key": bucket["key"], "count": bucket["doc_count"]}
            for bucket in aggs.get("procurement_methods", {}).get("buckets", [])
        ]
        statuses_agg = [
            {"key": bucket["key"], "count": bucket["doc_count"]}
            for bucket in aggs.get("contract_statuses", {}).get("buckets", [])
        ]

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "request": request,
            "q": q,
            "method": method,
            "status": status,
            "search_type": search_type,
            "page": page,
            "total_pages": total_pages,
            "total_value": total_value,
            "results": hits,
            "methods_agg": methods_agg,
            "statuses_agg": statuses_agg
                ,
                "rag_response": rag_response
        }
    )

@app.get("/contract/{ocid}", response_class=HTMLResponse)
async def contract_detail(request: Request, ocid: str):
    # Fetch exact contract matching the ocid
    query = {"query": {"term": {"ocid": ocid}}}
    try:
        response = es.search(index=INDEX_NAME, body=query, size=1)
        hits = response.get("hits", {}).get("hits", [])
        contract = hits[0]["_source"] if hits else None
    except Exception as e:
        print(f"Error fetching contract: {e}")
        contract = None

    return templates.TemplateResponse(
        request=request,
        name="detail.html",
        context={
            "request": request,
            "contract": contract,
            "ocid": ocid
        }
    )