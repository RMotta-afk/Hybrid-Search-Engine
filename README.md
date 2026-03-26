# Hybrid Search Engine

A FastAPI search API that combines **Pinecone vector search** with **BM25 keyword search** to deliver high-quality document retrieval. Supports weighted score fusion and Reciprocal Rank Fusion (RRF), with optional contextual retrieval enrichment via Gemini.

---

## Architecture

```
                Indexing Pipeline
                =================

POST /index ──> [chunk_document()] ──> [get_embeddings_batch()] ──> Pinecone
   { doc }              │               all-MiniLM-L6-v2 (384-dim)   (vectors)
                        └──────────────> [BM25Index.add_document()]
                                          (in-memory keyword index)


                Search Pipeline
                ===============

POST /search ──> [embed_query()] ──> Pinecone vector_search() ───┐
   { query }          │                                          │
                      └──────────> BM25Index.search() ───────────┤
                                                                  │
                                                            Merge & Rank
                                                    (weighted score or RRF)
                                                                  │
                                                                  v
                                                          Ranked Results
```

---

## Project Structure

```
Hybrid-Search-Engine/
├── app/
│   ├── config.py           # Settings via env vars (PINECONE_API_KEY, GOOGLE_API_KEY)
│   ├── chunking.py         # Sentence-boundary-aware document chunking (MD5 chunk IDs)
│   ├── bm25.py             # In-memory BM25Index (k1=1.5, b=0.75)
│   ├── embeddings.py       # all-MiniLM-L6-v2 batch/single embedding
│   ├── pinecone_client.py  # Pinecone setup & index_document()
│   ├── search.py           # vector_search, keyword_search, hybrid_search, rrf_search
│   ├── contextual.py       # Gemini chunk enrichment (optional)
│   └── main.py             # FastAPI app — /health, /index, /search
├── tests/
│   ├── test_chunking.py
│   ├── test_bm25.py
│   └── test_search.py
├── requirements.txt
└── .env.example
```

---

## Setup

### Prerequisites

- Python 3.11+
- [Pinecone](https://pinecone.io) account and API key (free tier provides 1 index)
- Google AI API key — optional, only needed for contextual retrieval

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

Key packages: `fastapi==0.115.12`, `uvicorn[standard]==0.34.2`, `pinecone==6.0.2`, `sentence-transformers==4.1.0`, `google-genai==1.14.0`

> The embedding model (`all-MiniLM-L6-v2`, ~80 MB) downloads automatically on first run.

### 2. Configure environment variables

```bash
cp .env.example .env
# Edit .env and fill in your keys
```

```
PINECONE_API_KEY=your-pinecone-api-key
GOOGLE_API_KEY=your-google-ai-key   # optional
```

### 3. Run the server

```bash
python -m app.main
```

Server starts at `http://localhost:8000`. On startup, the app automatically creates the Pinecone index `hybrid-search-lab` (dimension: 384, metric: cosine, serverless on AWS us-east-1) if it does not already exist.

---

## API

### `GET /health`

Returns server status and how many documents are currently in the BM25 index.

```json
{ "status": "ok", "bm25_docs": 42 }
```

---

### `POST /index`

Index a document. The document is split into overlapping chunks (default 500 chars, 100 overlap), embedded with `all-MiniLM-L6-v2`, upserted to Pinecone, and added to the in-memory BM25 index.

**Request**

```json
{
  "doc_id": "doc1",
  "title": "Introduction to Hybrid Search",
  "text": "Hybrid search combines the strengths of vector-based semantic search with traditional keyword-based search...",
  "contextual": false
}
```

Set `contextual: true` to enrich each chunk with a 1-2 sentence Gemini summary before indexing (requires `GOOGLE_API_KEY`). Uses `gemini-2.0-flash`.

**Response**

```json
{ "doc_id": "doc1", "chunks_indexed": 3, "total_chars": 412 }
```

**Example**

```bash
curl -X POST http://localhost:8000/index \
  -H "Content-Type: application/json" \
  -d '{
    "doc_id": "doc1",
    "title": "Introduction to Hybrid Search",
    "text": "Hybrid search combines the strengths of vector-based semantic search with traditional keyword-based search. Vector search excels at understanding meaning and context, while keyword search is better at exact term matching and handling rare terms. By combining both approaches with weighted scoring, hybrid search achieves better recall and precision than either method alone."
  }'
```

---

### `POST /search`

Search across indexed documents.

**Request**

```json
{
  "query": "how does semantic search work with keywords",
  "limit": 10,
  "vector_weight": 0.7,
  "bm25_weight": 0.3,
  "method": "weighted"
}
```

| Field | Default | Description |
|---|---|---|
| `query` | — | Natural language search query |
| `limit` | `10` | Number of results to return |
| `vector_weight` | `0.7` | Weight for vector scores (weighted method only) |
| `bm25_weight` | `0.3` | Weight for BM25 scores (weighted method only) |
| `method` | `"weighted"` | `"weighted"` or `"rrf"` |

**Response**

```json
{
  "results": [
    {
      "id": "doc1_a3f2...",
      "text": "Hybrid search combines...",
      "metadata": { "doc_id": "doc1", "title": "Introduction to Hybrid Search", "chunk_index": 0 },
      "hybrid_score": 0.87,
      "vector_score": 0.92,
      "bm25_score": 1.45
    }
  ],
  "count": 1
}
```

RRF results return `rrf_score` instead of `hybrid_score`, `vector_score`, and `bm25_score`.

**Examples**

```bash
# Weighted hybrid search
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "how does semantic search work with keywords", "limit": 5}'

# RRF search
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "vector and keyword combination", "limit": 5, "method": "rrf"}'
```

---

## Search Methods

### Weighted Score Fusion

Both score sets are min-max normalized to [0, 1], then combined:

```
hybrid_score = vector_weight * norm_vector_score + bm25_weight * norm_bm25_score
```

Each search fetches `limit × 3` candidates before merging. Best when you want to tune the balance between semantic and keyword relevance.

### Reciprocal Rank Fusion (RRF)

Uses only rank positions, not raw scores:

```
rrf_score += 1 / (k + rank + 1)    # k=60, rank is 0-based
```

Documents appearing in both result lists accumulate score from both. More robust when score distributions differ significantly between the two systems.

### When to use which

| Query type | Better method |
|---|---|
| Conceptual / paraphrased | Weighted (high vector weight) |
| Exact technical terms | Weighted (raise BM25 weight) or RRF |
| Mixed / unknown | RRF (rank-based, no tuning needed) |

---

## BM25 Index

Parameters: `k1=1.5`, `b=0.75`

Tokenization: lowercased `[a-z0-9]+` tokens with stop-word removal (`the`, `a`, `an`, `is`, `are`, `was`, `were`, `in`, `on`, `at`, `to`, `for`, `of`, `and`, `or`, `but`, `not`, `with`, `this`, `that`, `it`).

IDF formula: `log((N - df + 0.5) / (df + 0.5) + 1)`

> The BM25 index is **in-memory** and resets on server restart. Re-index documents after restarting.

---

## Running Tests

```bash
pytest tests/
```

- `test_chunking.py` — chunk boundaries, MD5 IDs, overlap, edge cases
- `test_bm25.py` — tokenization, scoring, ranking, stop words
- `test_search.py` — normalize_scores, hybrid fusion, RRF scoring (mocked Pinecone + real BM25)

---

## References

- [Pinecone Documentation](https://docs.pinecone.io/)
- [BM25 — Okapi BM25](https://en.wikipedia.org/wiki/Okapi_BM25)
- [Reciprocal Rank Fusion (paper)](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf)
- [Contextual Retrieval — Anthropic](https://www.anthropic.com/news/contextual-retrieval)
- [Sentence Transformers](https://www.sbert.net/)
