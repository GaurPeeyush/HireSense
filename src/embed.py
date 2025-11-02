from typing import Iterable, List
from openai import OpenAI
from .config import EMBEDDING_MODEL


_client = OpenAI()


def embed_texts(texts: Iterable[str]) -> List[List[float]]:
    batch = list(texts)
    if not batch:
        return []
    response = _client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
    return [item.embedding for item in response.data]


def embed_text(text: str) -> List[float]:
    return embed_texts([text])[0]


