from app.search import normalize_scores, hybrid_search, rrf_search


def test_normalize_scores_basic():
    results = [
        {"id": "a", "score": 10.0},
        {"id": "b", "score": 5.0},
        {"id": "c", "score": 0.0},
    ]
    normalized = normalize_scores(results, "score")
    assert normalized["a"] == 1.0
    assert normalized["c"] == 0.0
    assert normalized["b"] == 0.5


def test_normalize_scores_all_equal():
    results = [{"id": "a", "score": 3.0}, {"id": "b", "score": 3.0}]
    normalized = normalize_scores(results, "score")
    assert normalized["a"] == 1.0
    assert normalized["b"] == 1.0


def test_normalize_scores_empty():
    assert normalize_scores([], "score") == {}


def test_hybrid_score_computation():
    """Test hybrid_search merges and weights results correctly using mocks."""
    from unittest.mock import MagicMock, patch
    from app.bm25 import BM25Index

    mock_index = MagicMock()
    mock_index.query.return_value = MagicMock(matches=[
        MagicMock(id="doc1_chunk1", score=0.9, metadata={"text": "vector result 1", "doc_id": "doc1", "title": "T"}),
        MagicMock(id="doc2_chunk1", score=0.5, metadata={"text": "vector result 2", "doc_id": "doc2", "title": "T"}),
    ])

    bm25 = BM25Index()
    bm25.add_document("doc1_chunk1", "keyword result one python machine learning")
    bm25.add_document("doc3_chunk1", "keyword result two python data science")

    with patch("app.search.embed_query", return_value=[0.1] * 384):
        results = hybrid_search("python", mock_index, bm25, limit=10, vector_weight=0.7, bm25_weight=0.3)

    assert len(results) > 0
    for r in results:
        assert "hybrid_score" in r
        assert r["hybrid_score"] >= 0.0


def test_rrf_score_computation():
    """Test rrf_search computes RRF scores correctly using mocks."""
    from unittest.mock import MagicMock, patch
    from app.bm25 import BM25Index

    mock_index = MagicMock()
    mock_index.query.return_value = MagicMock(matches=[
        MagicMock(id="doc1_chunk1", score=0.9, metadata={"text": "rrf vector 1", "doc_id": "doc1", "title": "T"}),
        MagicMock(id="doc2_chunk1", score=0.6, metadata={"text": "rrf vector 2", "doc_id": "doc2", "title": "T"}),
    ])

    bm25 = BM25Index()
    bm25.add_document("doc1_chunk1", "rrf keyword doc one python")
    bm25.add_document("doc3_chunk1", "rrf keyword doc three python")

    with patch("app.search.embed_query", return_value=[0.1] * 384):
        results = rrf_search("python", mock_index, bm25, limit=10, k=60)

    assert len(results) > 0
    for r in results:
        assert "rrf_score" in r
        assert r["rrf_score"] > 0.0

    # doc1 appears in both lists → higher rrf score than doc2 (only vector)
    scores = {r["id"]: r["rrf_score"] for r in results}
    if "doc1_chunk1" in scores and "doc2_chunk1" in scores:
        assert scores["doc1_chunk1"] > scores["doc2_chunk1"]
