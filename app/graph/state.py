# app/graph/state.py
from typing import List
from typing_extensions import NotRequired, TypedDict
from langchain_core.documents import Document

class GraphState(TypedDict):
    question: str
    documents: List[Document]
    retry_count: int
    original_question: NotRequired[str]
    generation: NotRequired[str]