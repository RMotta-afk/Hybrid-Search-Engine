import time
from pinecone import Pinecone, ServerlessSpec
from app import config
from app.chunking import chunk_document
from app.embeddings import get_embeddings_batch

index = None


def get_pinecone_client() -> Pinecone:
    return Pinecone(api_key=config.PINECONE_API_KEY)


def setup_index():
    pc = get_pinecone_client()
    existing = [idx.name for idx in pc.list_indexes()]

    if config.INDEX_NAME not in existing:
        pc.create_index(
            name=config.INDEX_NAME,
            dimension=config.EMBEDDING_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )

    while not pc.describe_index(config.INDEX_NAME).status.ready:
        time.sleep(1)

    return pc.Index(config.INDEX_NAME)


def index_document(doc_id: str, title: str, text: str, index, bm25_index) -> dict:
    chunks = chunk_document(text)
    chunk_texts = [c["text"] for c in chunks]

    all_embeddings: list = []
    batch_size = 64
    for i in range(0, len(chunk_texts), batch_size):
        batch = chunk_texts[i:i + batch_size]
        all_embeddings.extend(get_embeddings_batch(batch))

    vectors = []
    for chunk, embedding in zip(chunks, all_embeddings):
        vectors.append({
            "id": f"{doc_id}_{chunk['id']}",
            "values": embedding,
            "metadata": {
                "doc_id": doc_id,
                "title": title,
                "text": chunk["text"],
                "chunk_index": chunk["chunk_index"],
            },
        })

    upsert_batch_size = 100
    for i in range(0, len(vectors), upsert_batch_size):
        index.upsert(vectors=vectors[i:i + upsert_batch_size])

    for chunk in chunks:
        bm25_index.add_document(
            f"{doc_id}_{chunk['id']}",
            chunk["text"],
            {"doc_id": doc_id, "title": title, "chunk_index": chunk["chunk_index"]},
        )

    return {"doc_id": doc_id, "chunks_indexed": len(chunks), "total_chars": len(text)}
