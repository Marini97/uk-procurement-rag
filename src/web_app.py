from fastapi import FastAPI, Request, Query
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from src.es_client import get_es_client, search_contracts, INDEX_NAME
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
        response = {"hits": {"total": {"value": 0}, "hits": []}, "aggregations": {}}
        print(f"Error during search: {e}")

    # Extract hits
    hits = response.get("hits", {}).get("hits", [])
    
    # Extract total based on ES version mapping (total can be an int or a dict)
    total_obj = response.get("hits", {}).get("total", {})
    if isinstance(total_obj, dict):
        total_value = total_obj.get("value", 0)
    else:
        total_value = total_obj
        
    # Calculate pages
    total_pages = math.ceil(total_value / size) if total_value > 0 else 0
    total_pages = min(total_pages, 100) # limit to max 100 pages for safety

    # Extract aggregations for filters
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