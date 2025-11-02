## Resume QA with Category Filtering, Pinecone, and OpenAI

### Overview
This project builds a simple retrieval-augmented QA system over resumes. Each CSV row is a single document stored in Pinecone along with metadata. At query time, we:
- Classify the query into one category (via OpenAI),
- Embed the query (OpenAI embeddings),
- Metadata-filter by the predicted category and run vector search in Pinecone,
- Return the top-5 documents as-is and generate a readable answer with an LLM.

The Streamlit UI shows both the answer and the documents, along with a default-open trace of the entire pipeline so reviewers can see each step.

### Stack
- OpenAI Embeddings: `text-embedding-3-small` (1536-dim, cosine)
- OpenAI Chat: `gpt-4o-mini` (adjustable)
- Vector DB: Pinecone (serverless)
- Orchestration/UI: Streamlit
- Validation: Pydantic (category tool)

## Sample Conversations

<img width="1918" height="1015" alt="Screenshot 2025-11-02 at 5 25 57 PM" src="https://github.com/user-attachments/assets/eb44a005-5615-4164-b2a0-cfb52d7cf1f5" />
<img width="1918" height="1015" alt="Screenshot 2025-11-02 at 6 07 03 PM" src="https://github.com/user-attachments/assets/c11f284a-cdc2-469c-affb-8b39585654d1" />
<img width="1918" height="1015" alt="Screenshot 2025-11-02 at 6 06 34 PM" src="https://github.com/user-attachments/assets/fed321da-e4f3-4173-9abc-4c7b3c7a2f97" />
<img width="1918" height="1015" alt="Screenshot 2025-11-02 at 5 28 05 PM" src="https://github.com/user-attachments/assets/7d072782-5b7e-46f9-99f0-867512a71e60" />
<img width="1918" height="1015" alt="Screenshot 2025-11-02 at 5 27 04 PM" src="https://github.com/user-attachments/assets/437467ae-208e-4d76-b378-2323c1174188" />

## Quickstart

### 1) Environment
Create `.env` in the project root:
```bash
cat > .env << 'EOF'
OPENAI_API_KEY=YOUR_OPENAI_KEY
PINECONE_API_KEY=YOUR_PINECONE_KEY
PINECONE_INDEX=resumes-index
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1
EOF
```

### 2) Install dependencies
```bash
pip install -r requirements.txt
```

### 3) Upsert data (one-time)
Your cleaned CSV should have at least the columns `Resume` and `Category` (and optionally `id`). Example path used here:
```bash
python upsert_resumes.py \
  --csv Resume_cleaned.csv
```
What this does:
- Ensures a Pinecone index (cosine, dim=1536)
- Embeds `Resume` text
- Upserts each row with: `id`, `values` (vector), `metadata` = `{ category, text, row_id }`

### 4) Run the app
```bash
streamlit run app.py
```

---

## Data assumptions
- Each row is a separate document (no chunking).
- Required columns: `Resume`, `Category`.
- Optional: `id` column; if missing, IDs are generated as `row_{i}`.
- Categories come from a fixed set used for metadata filtering:
  - HR, DESIGNER, INFORMATION-TECHNOLOGY, TEACHER, ADVOCATE, BUSINESS-DEVELOPMENT, HEALTHCARE, FITNESS, AGRICULTURE, BPO, SALES, CONSULTANT, DIGITAL-MEDIA, AUTOMOBILE, CHEF, FINANCE, APPAREL, ENGINEERING, ACCOUNTANT, CONSTRUCTION, PUBLIC-RELATIONS, BANKING, ARTS, AVIATION.

---

## How it works (end-to-end)
1) Category classification (OpenAI chat)
- Given the user’s question, the system returns exactly one category from the fixed enum using a Pydantic-validated JSON response.

2) Query embedding (OpenAI embeddings)
- Embeds the user’s question with `text-embedding-3-small` (1536-dim).

3) Metadata filter + vector search (Pinecone)
- Queries the index with filter `{ "category": <chosen_category> }` and metric = cosine.
- Retrieves top-5 documents, including IDs, scores, and stored text in metadata.

4) LLM answer generation (OpenAI chat)
- Sends the exact user question and the 5 documents to the LLM to produce a concise, readable answer. IDs can be referenced inline.

5) UI display (Streamlit)
- Left column: LLM answer
- Right column: Top 5 documents “as-is” with IDs and scores
- Default-open trace expander shows each step: tools, models, filters, durations, fetched IDs.

---

## Repository layout
```text
  README.md                  # This file
  requirements.txt           # Python deps
  .env                       # Your keys (not committed)
  app.py                     # Streamlit UI
  scripts/
    upsert_resumes.py        # One-time embed & upsert from CSV to Pinecone
  src/
    config.py                # Env loading, constants (embedding model, categories, index)
    embed.py                 # OpenAI embedding helpers
    pinecone_client.py       # Pinecone client + index helpers
    tools.py                 # Pydantic enum + category classifier (OpenAI chat)
    llm.py                   # LLM answer generation (+trace)
    query_pipeline.py        # Retrieval+filtering+search and answer orchestration (+trace)
  Resume/
    Resume_cleaned.csv       # Your cleaned dataset
```

---

## Key files explained

### `src/config.py`
- Loads env vars and centralizes constants:
  - Embedding model: `text-embedding-3-small` (dim = 1536)
  - Category list (enum)
  - Pinecone index name, cloud, region

### `src/embed.py`
- `embed_text(text)`, `embed_texts(texts)`: wrappers around OpenAI Embeddings API.

### `src/pinecone_client.py`
- `ensure_index(name, dimension)`: creates a serverless index if missing.
- `get_index(name)`: returns the Pinecone index handle.

### `scripts/upsert_resumes.py`
- Reads the CSV, ensures columns, generates/uses IDs, embeds `Resume`, and upserts documents to Pinecone with metadata `{ category, text, row_id }`.

### `src/tools.py`
- `CategoryEnum` with the fixed allowed categories.
- `classify_category(user_query)`: OpenAI chat call that returns a JSON object parsed by Pydantic to ensure exactly one valid category.

### `src/query_pipeline.py`
- `retrieve(user_query, top_k=5)`: classify → embed → Pinecone query with metadata filter; returns `{ category, docs, trace }`.
- `answer(user_query, retrieved)`: generates a final answer from the top-5 docs; returns `{ answer, trace }`.

### `src/llm.py`
- `generate_answer(...)`: calls OpenAI chat to produce a readable answer using the 5 docs.
- `generate_answer_with_trace(...)`: same but also returns a small trace object for the UI.

### `app.py` (Streamlit)
- Simple UI with:
  - Category badge
  - Two columns for answer and documents
  - Default-open “Processing Trace” with tools, models, filters, durations, and the 5 IDs

---

## Switch models / tune behavior
- Change embedding model: edit `EMBEDDING_MODEL` in `src/config.py` and ensure the dimension matches the index.
- Change chat model: update `model` in `src/tools.py` and `src/llm.py`.
- Change topK: pass a different `top_k` to `retrieve()` or adjust default in `src/query_pipeline.py`.
- Change index name/region/cloud: update `.env` and/or `src/config.py`.

---

## Troubleshooting
- Missing env var: `Missing required env var: OPENAI_API_KEY`
  - Ensure `.env` exists in the project root with your keys.

- “No documents found”
  - Confirm the CSV categories match the fixed enum exactly.
  - Ensure the upsert completed without errors and targeted the same index configured at runtime.

- Costs and rate limits
  - Embeddings and chat both call OpenAI APIs; Pinecone calls are billed per capacity/usage. Keep topK small (5) for demos.

---

## Example commands
```bash
# Upsert
python upsert_resumes.py \
  --csv Resume_cleaned.csv

# Run UI
streamlit run app.py
```

---


