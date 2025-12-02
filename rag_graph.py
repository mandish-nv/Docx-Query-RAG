# rag_graph.py
import time
from langgraph.graph import StateGraph, END
from dataclasses import dataclass
from typing import List, Dict, Any
import rag_query


@dataclass
class RAGState:
    user_query: str
    refined_query: str | None = None
    retrieved_docs: List[Dict[str, Any]] | None = None
    answer: str | None = None
    chat_history: list | None = None
    timings: Dict[str, float] = None


# -------------------- NODES --------------------

def refine_query_node(state: dict) -> dict:
    start = time.time()
    
    refined = rag_query.generate_refined_query(state["user_query"])
    state["refined_query"] = refined

    state["timings"]["refine_query_ms"] = (time.time() - start) * 1000
    return state


def retrieve_docs_node(state: dict) -> dict:
    start = time.time()

    answer, docs = rag_query.query_qdrant_rag(
        user_query=state["refined_query"],
        chat_history=state["chat_history"] or [],
    )

    state["answer"] = answer
    state["retrieved_docs"] = docs

    state["timings"]["retrieve_docs_ms"] = (time.time() - start) * 1000
    return state


def final_answer_node(state: dict) -> dict:
    # Just record timing
    state["timings"]["total_ms"] = (
        state["timings"]["refine_query_ms"]
        + state["timings"]["retrieve_docs_ms"]
    )
    return state


# -------------------- BUILD GRAPH --------------------

def build_rag_graph():
    graph = StateGraph(dict)

    graph.add_node("refine_query", refine_query_node)
    graph.add_node("retrieve_docs", retrieve_docs_node)
    graph.add_node("final_answer", final_answer_node)

    graph.set_entry_point("refine_query")
    graph.add_edge("refine_query", "retrieve_docs")
    graph.add_edge("retrieve_docs", "final_answer")
    graph.add_edge("final_answer", END)

    return graph.compile()


rag_graph = build_rag_graph()
