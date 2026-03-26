from google import genai
from app import config
from app.chunking import chunk_document
from app.embeddings import get_embeddings_batch


def enrich_chunk_with_context(document_title: str, full_document: str, chunk: str) -> str:
    client = genai.Client(api_key=config.GOOGLE_API_KEY)
    prompt = (
        f"You are an assistant that provides context for document chunks.\n\n"
        f"Document title: {document_title}\n"
        f"Document excerpt (first 3000 chars): {full_document[:3000]}\n\n"
        f"Chunk: {chunk}\n\n"
        f"Write 1-2 sentences explaining what this chunk is about in the context of the overall document. "
        f"Be specific and concise. Return only the contextual prefix, nothing else."
    )
    response = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
    return f"{response.text.strip()}\n\n{chunk}"


def index_with_contextual_retrieval(
    doc_id: str, title: str, text: str, index, bm25_index
) -> dict:
    chunks = chunk_document(text)
    enriched_texts = [enrich_chunk_with_context(title, text, c["text"]) for c in chunks]

    all_embeddings: list = []
    batch_size = 64
    for i in range(0, len(enriched_texts), batch_size):
        batch = enriched_texts[i:i + batch_size]
        all_embeddings.extend(get_embeddings_batch(batch))

    vectors = []
    for chunk, enriched, embedding in zip(chunks, enriched_texts, all_embeddings):
        vectors.append({
            "id": f"{doc_id}_{chunk['id']}",
            "values": embedding,
            "metadata": {
                "doc_id": doc_id,
                "title": title,
                "text": enriched,
                "chunk_index": chunk["chunk_index"],
            },
        })

    upsert_batch_size = 100
    for i in range(0, len(vectors), upsert_batch_size):
        index.upsert(vectors=vectors[i:i + upsert_batch_size])

    for chunk, enriched in zip(chunks, enriched_texts):
        bm25_index.add_document(
            f"{doc_id}_{chunk['id']}",
            enriched,
            {"doc_id": doc_id, "title": title, "chunk_index": chunk["chunk_index"]},
        )

    return {"doc_id": doc_id, "chunks_indexed": len(chunks), "total_chars": len(text)}
