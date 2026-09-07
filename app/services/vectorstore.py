# app/services/vectorstore.py
import os
from typing import Optional
from langchain_ollama import OllamaEmbeddings
from langchain_milvus import Milvus
from langchain_core.vectorstores import VectorStoreRetriever
from app.config import settings

_embeddings: Optional[OllamaEmbeddings] = None
_vectorstore: Optional[Milvus] = None
_retriever: Optional[VectorStoreRetriever] = None

def get_embeddings() -> OllamaEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = OllamaEmbeddings(
            model=settings.EMBEDDING_MODEL,
            base_url=settings.OLLAMA_BASE_URL,
        )
    return _embeddings

def get_vectorstore() -> Milvus:
    global _vectorstore
    if _vectorstore is None:
        db_path = os.path.abspath(settings.MILVUS_DB_PATH)
        parent_dir = os.path.dirname(db_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

        _vectorstore = Milvus(
            embedding_function=get_embeddings(),
            connection_args={"uri": db_path},
            collection_name=settings.COLLECTION_NAME,
            auto_id=True,
        )
    return _vectorstore

def get_retriever() -> VectorStoreRetriever:
    global _retriever
    if _retriever is None:
        _retriever = get_vectorstore().as_retriever(search_kwargs={"k": 2})
    return _retriever