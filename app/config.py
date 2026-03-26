import os
from dotenv import load_dotenv

load_dotenv()

PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY", "")
GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
INDEX_NAME: str = "hybrid-search-lab"
EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
EMBEDDING_DIM: int = 384
