# app/config.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "Local Agentic RAG Engine"
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    LLM_MODEL: str = "llama3.2"
    EMBEDDING_MODEL: str = "nomic-embed-text"
    MILVUS_DB_PATH: str = "./data/milvus_demo.db"
    COLLECTION_NAME: str = "rag_knowledge_base"
    MAX_RETRIES: int = 2

settings = Settings()