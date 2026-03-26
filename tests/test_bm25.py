from app.bm25 import BM25Index


def test_tokenize_lowercases_and_removes_stop_words():
    tokens = BM25Index.tokenize("The quick brown fox is fast")
    assert "the" not in tokens
    assert "is" not in tokens
    assert "quick" in tokens
    assert "brown" in tokens
    assert "fox" in tokens


def test_tokenize_extracts_alphanumeric():
    tokens = BM25Index.tokenize("hello-world foo123 bar!")
    assert "hello" in tokens
    assert "world" in tokens
    assert "foo123" in tokens
    assert "bar" in tokens


def test_add_document_updates_state():
    idx = BM25Index()
    idx.add_document("doc1", "machine learning algorithms", {"title": "ML"})
    assert idx.total_docs == 1
    assert "doc1" in idx.documents
    assert idx.doc_lengths["doc1"] > 0
    assert idx.avg_doc_length > 0


def test_add_multiple_documents_updates_df():
    idx = BM25Index()
    idx.add_document("doc1", "python programming language")
    idx.add_document("doc2", "python data science")
    assert idx.df.get("python", 0) == 2
    assert idx.total_docs == 2


def test_search_returns_relevant_docs():
    idx = BM25Index()
    idx.add_document("doc1", "machine learning deep neural networks")
    idx.add_document("doc2", "cooking recipes italian pasta")
    idx.add_document("doc3", "machine learning random forest")
    results = idx.search("machine learning")
    ids = [r["id"] for r in results]
    assert "doc1" in ids
    assert "doc3" in ids
    assert "doc2" not in ids


def test_search_nonexistent_term_returns_empty():
    idx = BM25Index()
    idx.add_document("doc1", "hello world foo bar")
    results = idx.search("zzz_nonexistent_term_xyz")
    assert results == []


def test_search_scores_are_positive():
    idx = BM25Index()
    idx.add_document("doc1", "quantum computing qubits superposition")
    idx.add_document("doc2", "quantum entanglement physics")
    results = idx.search("quantum computing")
    for r in results:
        assert r["score"] > 0


def test_search_respects_limit():
    idx = BM25Index()
    for i in range(20):
        idx.add_document(f"doc{i}", f"search query term document {i}")
    results = idx.search("search query term", limit=5)
    assert len(results) <= 5
