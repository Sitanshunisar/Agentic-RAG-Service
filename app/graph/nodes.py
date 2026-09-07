# app/graph/nodes.py
from langchain_core.prompts import ChatPromptTemplate
from app.graph.state import GraphState
from app.services.llm import GradeDocuments, get_document_grader, get_llm
from app.services.vectorstore import get_retriever

grader = get_document_grader()
llm = get_llm()

retriever = get_retriever()

def retrieve_node(state: GraphState) -> dict:
    query = state.get("question", "")
    print(f"\n[NODE: retrieve] Querying with: '{query}'")
    documents = retriever.invoke(query)
    print(f"[NODE: retrieve] Documents retrieved: {len(documents)}")
    return {
        "documents": documents,
        "retry_count": state.get("retry_count", 0),
    }

# In app/graph/nodes.py

def grade_documents_node(state: GraphState) -> dict:
    question = state.get("original_question") or state.get("question", "")
    documents = state.get("documents", [])
    print(f"\n[NODE: grade_documents] Grader evaluating {len(documents)} docs for query: '{question}'")

    grader_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are an evaluator assessing if a document contains any relevant information, keywords, or context to answer a question.\n"
            "Rules:\n"
            "- If the document shares keywords, entities (like names or projects), or concepts with the question, reply 'yes'.\n"
            "- If the document has zero connection to the question, reply 'no'.\n"
            "- Reply ONLY with the single word 'yes' or 'no'."
        ),
        (
            "human",
            "Question: {question}\n\nDocument:\n{document}\n\nRelevant (yes/no):"
        )
    ])

    # Direct LLM invocation is faster and more reliable on llama3.2 than structured grammar
    chain = grader_prompt | llm

    graded_docs = []
    for idx, doc in enumerate(documents):
        try:
            response = chain.invoke(
                {"question": question, "document": doc.page_content}
            )
            raw_text = str(response.content).strip().lower()
            
            # Extract score cleanly
            is_relevant = "yes" in raw_text or raw_text.startswith("y")
            print(f"[NODE: grade_documents] Doc {idx + 1} raw response: '{raw_text}' -> Match: {is_relevant}")

            if is_relevant:
                graded_docs.append(doc)
        except Exception as e:
            print(f"[!] Grader error on doc {idx + 1}: {e}. Retaining as fallback.")
            graded_docs.append(doc)

    if not graded_docs and documents:
        print("[NODE: grade_documents] Warning: All docs filtered by grader. Passing retrieved docs as fallback.")
        graded_docs = documents

    print(f"[NODE: grade_documents] Passing forward {len(graded_docs)} doc(s)")
    return {"documents": graded_docs}

def transform_query_node(state: GraphState) -> dict:
    original_q = state.get("original_question") or state.get("question", "")
    print(f"\n[NODE: transform_query] Rewriting query: '{original_q}'")

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are an expert search-query optimizer. Output ONLY a concise search phrase (3 to 6 words) optimized for semantic vector lookup. "
            "Do NOT include explanations, quotes, or boolean operators."
        ),
        ("human", "Original user question: {question}\n\nSearch keywords:")
    ])

    chain = prompt | llm
    new_query = chain.invoke({"question": original_q}).content
    clean_query = str(new_query).strip().replace('"', '').replace("`", "")
    print(f"[NODE: transform_query] New search query: '{clean_query}'")

    return {
        "question": clean_query,
        "retry_count": state.get("retry_count", 0) + 1,
    }

def generate_node(state: GraphState) -> dict:
    user_question = state.get("original_question") or state.get("question", "")
    documents = state.get("documents", [])
    print(f"\n[NODE: generate] Generating response using {len(documents)} doc(s)")

    context_str = "\n\n".join(doc.page_content for doc in documents)
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a factual QA assistant. Answer the user's question using ONLY the provided document context. "
            "Be direct and concise. If the context does not contain the answer or is empty, say 'I don\\'t know.'.\n\n"
            "Document Context:\n{context}"
        ),
        ("human", "{question}")
    ])

    chain = prompt | llm
    response = chain.invoke({"context": context_str, "question": user_question})
    return {"generation": str(response.content).strip()}