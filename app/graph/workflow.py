from langgraph.graph import StateGraph, END
from app.graph.state import GraphState
from app.graph.nodes import retrieve_node, grade_documents_node, transform_query_node, generate_node
from app.graph.edges import decide_to_generate

def create_agent_graph():
    workflow = StateGraph(GraphState)

    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("grade_documents", grade_documents_node)
    workflow.add_node("transform_query", transform_query_node)
    workflow.add_node("generate", generate_node)

    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "transform_query": "transform_query",
            "generate": "generate",
        }
    )
    workflow.add_edge("transform_query", "retrieve")
    workflow.add_edge("generate", END)

    return workflow.compile()

compiled_graph = create_agent_graph()