from sentence_transformers import SentenceTransformer
from app import config

model = SentenceTransformer(config.EMBEDDING_MODEL)


def get_embeddings_batch(texts: list[str]) -> list[list[float]]:
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return [e.tolist() for e in embeddings]


def embed_query(query: str) -> list[float]:
    return get_embeddings_batch([query])[0]
