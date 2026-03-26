from app.embeddings import embed_query
from app.bm25 import BM25Index


def vector_search(query_embedding: list[float], index, limit: int = 20) -> list[dict]:
    results = index.query(vector=query_embedding, top_k=limit, include_metadata=True)
    return [
        {
            "id": m.id,
            "text": m.metadata["text"],
            "metadata": m.metadata,
            "vector_score": m.score,
        }
        for m in results.matches
    ]


def keyword_search(query: str, bm25_index: BM25Index, limit: int = 20) -> list[dict]:
    results = bm25_index.search(query, limit)
    return [
        {
            "id": r["id"],
            "text": r["text"],
            "metadata": r["metadata"],
            "bm25_score": r["score"],
        }
        for r in results
    ]


def normalize_scores(results: list[dict], score_key: str) -> dict[str, float]:
    if not results:
        return {}
    scores = [r[score_key] for r in results]
    min_score = min(scores)
    max_score = max(scores)
    if max_score == min_score:
        return {r["id"]: 1.0 for r in results}
    return {r["id"]: (r[score_key] - min_score) / (max_score - min_score) for r in results}


def hybrid_search(
    query: str,
    index,
    bm25_index: BM25Index,
    limit: int = 10,
    vector_weight: float = 0.7,
    bm25_weight: float = 0.3,
) -> list[dict]:
    query_embedding = embed_query(query)
    vec_results = vector_search(query_embedding, index, limit=limit * 3)
    kw_results = keyword_search(query, bm25_index, limit=limit * 3)

    norm_vec = normalize_scores(vec_results, "vector_score")
    norm_bm25 = normalize_scores(kw_results, "bm25_score")

    # Build lookup dicts for text/metadata
    lookup: dict = {}
    for r in vec_results:
        lookup[r["id"]] = {"text": r["text"], "metadata": r["metadata"],
                           "vector_score": r["vector_score"], "bm25_score": 0.0}
    for r in kw_results:
        if r["id"] in lookup:
            lookup[r["id"]]["bm25_score"] = r["bm25_score"]
        else:
            lookup[r["id"]] = {"text": r["text"], "metadata": r["metadata"],
                                "vector_score": 0.0, "bm25_score": r["bm25_score"]}

    all_ids = set(norm_vec.keys()) | set(norm_bm25.keys())
    scored = []
    for doc_id in all_ids:
        hybrid_score = vector_weight * norm_vec.get(doc_id, 0.0) + bm25_weight * norm_bm25.get(doc_id, 0.0)
        info = lookup[doc_id]
        scored.append({
            "id": doc_id,
            "text": info["text"],
            "metadata": info["metadata"],
            "hybrid_score": hybrid_score,
            "vector_score": info["vector_score"],
            "bm25_score": info["bm25_score"],
        })

    scored.sort(key=lambda x: x["hybrid_score"], reverse=True)
    return scored[:limit]


def rrf_search(
    query: str,
    index,
    bm25_index: BM25Index,
    limit: int = 10,
    k: int = 60,
) -> list[dict]:
    query_embedding = embed_query(query)
    vec_results = vector_search(query_embedding, index, limit=limit * 3)
    kw_results = keyword_search(query, bm25_index, limit=limit * 3)

    rrf_scores: dict = {}
    lookup: dict = {}

    for i, r in enumerate(vec_results):
        rrf_scores[r["id"]] = rrf_scores.get(r["id"], 0.0) + 1 / (k + i + 1)
        lookup[r["id"]] = {"text": r["text"], "metadata": r["metadata"]}

    for i, r in enumerate(kw_results):
        rrf_scores[r["id"]] = rrf_scores.get(r["id"], 0.0) + 1 / (k + i + 1)
        if r["id"] not in lookup:
            lookup[r["id"]] = {"text": r["text"], "metadata": r["metadata"]}

    ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:limit]
    return [
        {
            "id": doc_id,
            "text": lookup[doc_id]["text"],
            "metadata": lookup[doc_id]["metadata"],
            "rrf_score": score,
        }
        for doc_id, score in ranked
    ]
