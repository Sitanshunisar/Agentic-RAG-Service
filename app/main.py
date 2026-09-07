import os
import json
from typing import List

# Must be set before Milvus / gRPC modules load
os.environ["GRPC_ARG_KEEPALIVE_TIME_MS"] = "300000"
os.environ["GRPC_ARG_KEEPALIVE_TIMEOUT_MS"] = "20000"
os.environ["GRPC_ARG_HTTP2_MAX_PINGS_WITHOUT_DATA"] = "0"
os.environ["GRPC_ARG_KEEPALIVE_PERMIT_WITHOUT_CALLS"] = "0"
os.environ["GRPC_ARG_MIN_RECV_PING_INTERVAL_WITHOUT_DATA_MS"] = "300000"

from fastapi import FastAPI
from fastapi.responses import RedirectResponse, StreamingResponse
from langchain_core.documents import Document
from pydantic import BaseModel
from app.config import settings
from app.graph.state import GraphState
from app.graph.workflow import compiled_graph

app = FastAPI(title=settings.PROJECT_NAME)

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    source_documents: List[str]


# app/main.py

# app/main.py

def build_initial_state(query: str) -> GraphState:
    return {
        "original_question": query,
        "question": query,
        "documents": [],
        "retry_count": 0,
        "generation": "",
    }

@app.get("/")
def root():
    return RedirectResponse(url="/docs")

@app.get("/health")
def health_check():
    return {"status": "ok", "service": settings.PROJECT_NAME}

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(req: QueryRequest):
    result = await compiled_graph.ainvoke(build_initial_state(req.query))
    return QueryResponse(
        answer=result.get("generation", "No answer could be generated."),
        source_documents=[d.page_content for d in result.get("documents", [])]
    )

@app.post("/query/stream")
async def stream_query_endpoint(req: QueryRequest):
    initial_state = build_initial_state(req.query)

    async def event_generator():
        async for output in compiled_graph.astream(initial_state):
            for node_name, state_update in output.items():
                payload = {"node": node_name, "update": state_update}
                yield f"data: {json.dumps(payload, default=str)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")