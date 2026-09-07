# Local Agentic RAG Service

A high-performance, self-hosted **Agentic Retrieval-Augmented Generation (RAG)** system built with **FastAPI**, **LangGraph**, **Milvus Lite**, and **Ollama**.

The engine implements a **Corrective RAG (CRAG)** state machine to evaluate retrieved document chunks for semantic relevance, dynamically rewrite queries when context confidence is low, and synthesize grounded responses with **zero third-party cloud API dependencies or containerization overhead**.

---

## Key Highlights

* **Bare-Metal Local Inference**
  Runs 100% offline using Ollama:

  * `llama3.2` for generation and document grading
  * `nomic-embed-text` for embeddings

* **Embedded Milvus Lite Engine**
  Avoids heavy external vector database clusters and multi-gigabyte Docker setups by embedding storage directly on local disk.

* **Corrective RAG (CRAG) Control Flow**

  The system follows a corrective retrieval pipeline:

  1. **Retrieve** — Fetches the top-$k$ semantic chunks matching the user query.
  2. **Grade Documents** — Evaluates the relevance of retrieved chunks and rejects irrelevant noise.
  3. **Query Transformation** — Formulates optimized semantic queries when the initial relevance criteria are not satisfied.
  4. **Generate** — Synthesizes strictly grounded answers with built-in fallbacks to reduce hallucinations.

* **Optimized Transport Layer**
  Pre-configured HTTP/2 and gRPC keepalive parameters to reduce socket ping throttling on Windows systems.

* **Dual API Endpoints**

  * `POST /query` — Standard REST endpoint.
  * `POST /query/stream` — Real-time Server-Sent Events (SSE) endpoint for inspecting node transitions as the agent executes.

---

## Architecture

```text
                         ┌─────────────────┐
                         │   User Query    │
                         └────────┬────────┘
                                  │
                                  ▼
                         ┌─────────────────┐
                         │    Retrieve     │
                         └────────┬────────┘
                                  │
                                  ▼
                         ┌─────────────────┐
                         │ Grade Documents │
                         └────────┬────────┘
                                  │
                                  ▼
                    ┌────────────────────────────┐
                    │ Relevant Chunks Found?     │
                    └─────────────┬──────────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    │                           │
                   YES                         NO
                    │                    (Retry < MAX_RETRIES)
                    ▼                           │
             ┌─────────────┐                    ▼
             │   Generate  │          ┌─────────────────┐
             └──────┬──────┘          │ Transform Query │
                    │                 └────────┬────────┘
                    │                          │
                    │                          │
                    │                          └──────────────┐
                    │                                         │
                    │                                         ▼
                    │                                  ┌─────────────┐
                    │                                  │   Retrieve  │
                    │                                  └─────────────┘
                    │
                    ▼
             ┌───────────────┐
             │ Grounded      │
             │ Answer        │
             └───────────────┘
```

---

## Tech Stack

| Component               | Technology                                   |
| ----------------------- | -------------------------------------------- |
| **Framework**           | FastAPI, Uvicorn                             |
| **Agent Orchestration** | LangGraph, LangChain Core                    |
| **Vector Database**     | Milvus Lite (`pymilvus`, `langchain-milvus`) |
| **Local Models**        | Ollama (`llama3.2`, `nomic-embed-text`)      |
| **Inference**           | Local / Offline                              |
| **API**                 | REST + Server-Sent Events (SSE)              |
| **Target OS**           | Windows bare-metal                           |
| **Python**              | Python 3.11+                                 |

---

## Data Flow

```text
User Query
    │
    ▼
Query Embedding
    │
    ▼
Milvus Lite Vector Search
    │
    ▼
Top-k Document Chunks
    │
    ▼
LLM Relevance Grading
    │
    ├── Relevant ──────────────► Generate
    │                              │
    │                              ▼
    │                        Grounded Answer
    │
    └── Not Relevant
             │
             ▼
       Query Transformation
             │
             ▼
          Re-retrieve
             │
             ▼
          Re-grade
             │
             ▼
          Generate
```

---

## Project Goals

The service is designed to provide:

* **Fully local inference** without dependency on external LLM APIs.
* **Agentic retrieval correction** instead of relying solely on the first vector-search result.
* **Grounded generation** based on retrieved knowledge.
* **Low infrastructure overhead** through Milvus Lite.
* **Observable execution** through the streaming API.
* **Windows-friendly deployment** without requiring Docker.

---

## Local Models

### Generation & Document Grading

```text
llama3.2
```

Used for:

* Answer generation
* Document relevance grading
* Query transformation

### Embeddings

```text
nomic-embed-text
```

Used for:

* Converting documents into vector embeddings
* Converting user queries into vector embeddings
* Semantic similarity search through Milvus Lite

---

## API Endpoints

### Standard Query

```http
POST /query
```

Returns the final grounded response after the CRAG workflow completes.

### Streaming Query

```http
POST /query/stream
```

Returns Server-Sent Events (SSE) representing the agent's execution flow, including transitions between:

```text
Retrieve
   ↓
Grade Documents
   ↓
Transform Query (if required)
   ↓
Retrieve
   ↓
Generate
```

This endpoint is particularly useful for **debugging, demonstrations, and observing the agent's decision-making process**.

---

## Deployment Model

The system is designed to run directly on a Windows machine:

```text
┌─────────────────────────────────────────────┐
│              Windows Host                   │
│                                             │
│  ┌──────────────┐       ┌───────────────┐  │
│  │   FastAPI    │──────►│    Ollama     │  │
│  │   +          │       │               │  │
│  │  LangGraph   │       │   llama3.2    │  │
│  └──────┬───────┘       │   nomic       │  │
│         │               └───────────────┘  │
│         │                                   │
│         ▼                                   │
│  ┌──────────────┐                           │
│  │  Milvus Lite │                           │
│  │  Local Disk  │                           │
│  └──────────────┘                           │
│                                             │
└─────────────────────────────────────────────┘
```

### Requirements

* **Windows**
* **Python 3.11+**
* **Ollama**
* `llama3.2`
* `nomic-embed-text`

No external cloud LLM API or Docker deployment is required.

---

## Summary

**Local Agentic RAG Service** combines local LLM inference, semantic vector retrieval, and corrective agentic reasoning into a lightweight RAG architecture:

```text
                 ┌─────────────────────┐
                 │     User Query      │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │    Vector Search    │
                 │     Milvus Lite     │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │  Relevance Grading  │
                 │       Ollama        │
                 └──────────┬──────────┘
                            │
                  ┌─────────┴─────────┐
                  │                   │
              Relevant            Irrelevant
                  │                   │
                  │                   ▼
                  │          ┌─────────────────┐
                  │          │ Query Transform │
                  │          └────────┬────────┘
                  │                   │
                  │                   └──────► Re-retrieve
                  │
                  ▼
          ┌─────────────────┐
          │ Answer Generate │
          │     Ollama      │
          └────────┬────────┘
                   │
                   ▼
          ┌─────────────────┐
          │ Grounded Answer │
          └─────────────────┘
```

The result is a **fully local, self-hosted, corrective RAG service** with no dependency on third-party cloud inference APIs.
