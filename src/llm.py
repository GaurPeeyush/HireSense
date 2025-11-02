from typing import Dict, List, Tuple

from openai import OpenAI


_client = OpenAI()


def generate_answer(user_query: str, docs_with_ids: List[Tuple[str, str]]) -> str:
    # docs_with_ids: list of (id, text)
    docs_formatted = "\n\n".join([f"ID: {doc_id}\n{doc_text}" for doc_id, doc_text in docs_with_ids])
    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert assistant. Answer the user's query using the provided documents. "
                "Be concise, accurate, and reference document IDs inline when citing (e.g., [ID: 123])."
            ),
        },
        {
            "role": "user",
            "content": (
                f"User Query:\n{user_query}\n\nContext Documents (top 5):\n{docs_formatted}\n\n"
                "Answer the user's question using the context above."
            ),
        },
    ]
    resp = _client.chat.completions.create(model="gpt-4o-mini", messages=messages, temperature=0.2)
    return resp.choices[0].message.content or ""


def generate_answer_with_trace(user_query: str, docs_with_ids: List[Tuple[str, str]]) -> Tuple[str, Dict]:
    answer_text = generate_answer(user_query, docs_with_ids)
    trace = {
        "step": "LLM answer generation",
        "tool": "OpenAI Chat Completions",
        "model": "gpt-4o-mini",
        "input_docs_count": len(docs_with_ids),
    }
    return answer_text, trace


