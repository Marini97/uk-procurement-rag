import os
import logging
from typing import List, Optional

from src.es_client import get_es_client, search_contracts

logger = logging.getLogger(__name__)


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

    prompt = (
        "You are an expert assistant for UK procurement data. Use the contextual documents below to answer the question.\n\n"
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
