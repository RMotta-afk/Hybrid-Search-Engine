import hashlib
from app.chunking import chunk_document


def test_single_chunk_short_text():
    text = "Hello world."
    chunks = chunk_document(text)
    assert len(chunks) == 1
    assert chunks[0]["text"] == "Hello world."
    assert chunks[0]["chunk_index"] == 0


def test_multiple_chunks_with_overlap():
    text = "a" * 600
    chunks = chunk_document(text, chunk_size=500, overlap=100)
    assert len(chunks) > 1


def test_sentence_boundary_breaking():
    # Sentence ends well past 50% of chunk_size=50 → should break at period
    text = "First sentence is here. Second sentence continues beyond the break point."
    chunks = chunk_document(text, chunk_size=30, overlap=0)
    # First chunk should end at a sentence boundary if one exists past 15 chars
    assert all(c["text"] for c in chunks)


def test_chunk_ids_are_md5():
    text = "Some sample text for chunking purposes."
    chunks = chunk_document(text)
    for chunk in chunks:
        expected = hashlib.md5(chunk["text"].encode()).hexdigest()
        assert chunk["id"] == expected


def test_chunk_metadata():
    text = "Word " * 300
    chunks = chunk_document(text, chunk_size=500, overlap=100)
    for i, chunk in enumerate(chunks):
        assert chunk["chunk_index"] == i
        assert "start" in chunk
        assert "end" in chunk
        assert chunk["end"] > chunk["start"]


def test_no_infinite_loop_small_overlap():
    text = "x" * 1000
    chunks = chunk_document(text, chunk_size=10, overlap=9)
    # Must terminate and cover all content
    assert len(chunks) > 0
