import argparse
import math
import os
import sys
from typing import Iterable, List

import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import EMBEDDING_DIM, PINECONE_INDEX  # noqa: E402
from src.embed import embed_texts  # noqa: E402
from src.pinecone_client import ensure_index, get_index  # noqa: E402


def chunked(seq: List, size: int) -> Iterable[List]:
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def main():
    parser = argparse.ArgumentParser(description="Upsert resumes into Pinecone.")
    parser.add_argument(
        "--csv",
        dest="csv_path",
        default="/Users/peeyushgaur/Desktop/deccan/Resume/Resume_cleaned.csv",
        help="Path to the cleaned resumes CSV",
    )
    parser.add_argument(
        "--index",
        dest="index_name",
        default=PINECONE_INDEX,
        help="Pinecone index name",
    )
    parser.add_argument(
        "--batch",
        dest="batch_size",
        type=int,
        default=100,
        help="Batch size for embedding and upsert",
    )
    args = parser.parse_args()

    csv_path = args.csv_path
    index_name = args.index_name
    batch_size = args.batch_size

    if not os.path.exists(csv_path):
        print(f"CSV not found: {csv_path}")
        sys.exit(1)

    print(f"Loading: {csv_path}")
    df = pd.read_csv(csv_path)

    # Normalize columns
    if "Resume" not in df.columns and "Resume_str" in df.columns:
        df = df.rename(columns={"Resume_str": "Resume"})

    if "Resume" not in df.columns:
        print("CSV missing 'Resume' column.")
        sys.exit(1)

    if "Category" not in df.columns:
        print("CSV missing 'Category' column.")
        sys.exit(1)

    # Create/generate IDs
    if "id" in df.columns:
        ids = df["id"].astype(str).fillna("").tolist()
    else:
        ids = [f"row_{i}" for i in range(len(df))]

    texts = df["Resume"].astype(str).fillna("").tolist()
    categories = df["Category"].astype(str).fillna("").tolist()

    print(f"Ensuring index '{index_name}' (dim={EMBEDDING_DIM}) exists...")
    ensure_index(index_name, EMBEDDING_DIM)
    index = get_index(index_name)

    total = len(df)
    num_batches = math.ceil(total / batch_size)
    print(f"Upserting {total} rows in {num_batches} batches...")

    for id_batch, text_batch, cat_batch in tqdm(
        zip(chunked(ids, batch_size), chunked(texts, batch_size), chunked(categories, batch_size)),
        total=num_batches,
        desc="Embedding + Upsert",
    ):
        vectors = embed_texts(text_batch)
        upsert_records = []
        for vid, vec, txt, cat in zip(id_batch, vectors, text_batch, cat_batch):
            metadata = {
                "category": cat,
                "text": txt,
                "row_id": vid,
            }
            upsert_records.append({"id": vid, "values": vec, "metadata": metadata})

        index.upsert(vectors=upsert_records)

    print("Done upserting.")


if __name__ == "__main__":
    main()


