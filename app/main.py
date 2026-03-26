from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel
from app import config
from app.pinecone_client import setup_index, index_document
from app.bm25 import BM25Index
from app.search import hybrid_search, rrf_search

pinecone_index = None
bm25_index: BM25Index = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pinecone_index, bm25_index
    pinecone_index = setup_index()
    bm25_index = BM25Index()
    yield


app = FastAPI(title="Hybrid Search Engine", lifespan=lifespan)


class IndexRequest(BaseModel):
    doc_id: str
    title: str
    text: str
    contextual: bool = False


class SearchRequest(BaseModel):
    query: str
    limit: int = 10
    vector_weight: float = 0.7
    bm25_weight: float = 0.3
    method: str = "weighted"


@app.get("/health")
def health():
    return {"status": "ok", "bm25_docs": bm25_index.total_docs}


@app.post("/index")
def index(request: IndexRequest):
    if request.contextual and config.GOOGLE_API_KEY:
        from app.contextual import index_with_contextual_retrieval
        return index_with_contextual_retrieval(
            request.doc_id, request.title, request.text, pinecone_index, bm25_index
        )
    return index_document(request.doc_id, request.title, request.text, pinecone_index, bm25_index)


@app.post("/search")
def search(request: SearchRequest):
    if request.method == "rrf":
        results = rrf_search(request.query, pinecone_index, bm25_index, request.limit)
    else:
        results = hybrid_search(
            request.query, pinecone_index, bm25_index,
            request.limit, request.vector_weight, request.bm25_weight
        )
    return {"results": results, "count": len(results)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
