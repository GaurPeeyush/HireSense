import time
from typing import Optional
from pinecone import Pinecone, ServerlessSpec
from .config import (
    PINECONE_API_KEY,
    PINECONE_INDEX,
    PINECONE_CLOUD,
    PINECONE_REGION,
    EMBEDDING_DIM,
)


def get_pc() -> Pinecone:
    return Pinecone(api_key=PINECONE_API_KEY)


def ensure_index(index_name: Optional[str] = None, dimension: int = EMBEDDING_DIM) -> None:
    index_name = index_name or PINECONE_INDEX
    pc = get_pc()
    existing = {idx["name"] for idx in pc.list_indexes()}
    if index_name in existing:
        return
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric="cosine",
        spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
    )
    # Wait briefly for index to be ready
    for _ in range(30):
        desc = pc.describe_index(index_name)
        if desc.get("status", {}).get("ready"):
            break
        time.sleep(2)


def get_index(index_name: Optional[str] = None):
    index_name = index_name or PINECONE_INDEX
    pc = get_pc()
    return pc.Index(index_name)


