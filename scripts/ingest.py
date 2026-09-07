import os
import argparse
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_milvus import Milvus
from app.config import settings

def run_ingest(file_path: str):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    print(f"[+] Reading {file_path}...")
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".txt") or file_path.endswith(".md"):
        loader = TextLoader(file_path, encoding="utf-8")
    else:
        raise ValueError("Unsupported file format. Provide a .pdf, .txt, or .md file.")

    raw_docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=40)
    chunks = splitter.split_documents(raw_docs)
    print(f"[+] Generated {len(chunks)} document chunks.")

    embeddings = OllamaEmbeddings(
        model=settings.EMBEDDING_MODEL,
        base_url=settings.OLLAMA_BASE_URL,
    )

    # Resolve to an absolute path so Milvus Lite treats it as a file, not a directory
    db_file_path = os.path.abspath(settings.MILVUS_DB_PATH)
    db_dir = os.path.dirname(db_file_path)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)

    print(f"[+] Indexing chunks into Milvus Lite database at '{db_file_path}'...")
    Milvus.from_documents(
        documents=chunks,
        embedding=embeddings,
        connection_args={"uri": db_file_path},
        collection_name=settings.COLLECTION_NAME,
        drop_old=True,  # Overwrites stale/empty collections cleanly
    )
    print("[✓] Ingestion complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest documents into local Milvus Lite")
    parser.add_argument("--file", type=str, required=True, help="Path to PDF or TXT file")
    args = parser.parse_args()
    run_ingest(args.file)