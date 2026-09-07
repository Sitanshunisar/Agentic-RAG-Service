from app.config import settings
from app.graph.state import GraphState

def decide_to_generate(state: GraphState) -> str:
    # If no relevant chunks survived grading, retry query rewriting up to MAX_RETRIES
    if not state.get("documents"):
        if state.get("retry_count", 0) >= settings.MAX_RETRIES:
            return "generate"
        return "transform_query"
    return "generate"