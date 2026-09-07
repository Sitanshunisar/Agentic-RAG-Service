from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field
from app.config import settings

class GradeDocuments(BaseModel):
    """Binary assessment score for retrieved document relevance."""
    binary_score: str = Field(
        description="Relevance judgment: 'yes' if document is relevant to question, 'no' otherwise"
    )

def get_llm():
    return ChatOllama(
        model=settings.LLM_MODEL,
        base_url=settings.OLLAMA_BASE_URL,
        temperature=0,
    )

def get_document_grader():
    llm = get_llm()
    return llm.with_structured_output(GradeDocuments)