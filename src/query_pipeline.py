import time
from typing import Dict, List, Tuple

from .embed import embed_text
from .llm import generate_answer_with_trace
from .pinecone_client import get_index
from .tools import classify_category
from .config import EMBEDDING_MODEL, EMBEDDING_DIM, PINECONE_INDEX


def retrieve(user_query: str, top_k: int = 5) -> Dict:
    trace: List[Dict] = []

    t0 = time.time()
    category = classify_category(user_query)
    t1 = time.time()
    trace.append(
        {
            "step": "Category classification",
            "tool": "OpenAI Chat Completions",
            "model": "gpt-4o-mini",
            "chosen_category": category,
            "duration_ms": int((t1 - t0) * 1000),
        }
    )

    t2 = time.time()
    query_vec = embed_text(user_query)
    t3 = time.time()
    trace.append(
        {
            "step": "Query embedding",
            "tool": "OpenAI Embeddings",
            "model": EMBEDDING_MODEL,
            "dimension": EMBEDDING_DIM,
            "duration_ms": int((t3 - t2) * 1000),
        }
    )
    index = get_index()
    pinecone_filter = {"category": category}
    t4 = time.time()
    result = index.query(
        vector=query_vec,
        top_k=top_k,
        include_metadata=True,
        filter=pinecone_filter,
    )
    t5 = time.time()
    trace.append(
        {
            "step": "Vector search",
            "tool": "Pinecone",
            "index": PINECONE_INDEX,
            "metric": "cosine",
            "top_k": top_k,
            "filter": pinecone_filter,
            "duration_ms": int((t5 - t4) * 1000),
        }
    )

    matches = result.get("matches", [])
    docs = []
    for m in matches:
        doc_id = m.get("id")
        meta = m.get("metadata", {}) or {}
        text = meta.get("text", "")
        score = m.get("score")
        docs.append({"id": doc_id, "text": text, "score": score, "category": meta.get("category")})

    trace.append(
        {
            "step": "Fetch top 5",
            "tool": "Pinecone",
            "result_count": len(docs),
            "ids": [d["id"] for d in docs[:5]],
        }
    )

    return {"category": category, "docs": docs, "trace": trace}


def answer(user_query: str, retrieved: Dict) -> Dict:
    docs = retrieved.get("docs", [])[:5]
    docs_with_ids: List[Tuple[str, str]] = [(d["id"], d["text"]) for d in docs]
    ans_text, ans_trace = generate_answer_with_trace(user_query, docs_with_ids)
    return {"answer": ans_text, "trace": ans_trace}


