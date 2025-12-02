# rag_query.py
import streamlit as st
import requests
import json
from qdrant_client import models
import config
import time

# ---------------------------------------
# Existing Functions (UNCHANGED)
# ---------------------------------------

def generate_refined_query(user_query):
    prompt = f"{config.QUERY_GEN_PROMPT}\nUser Question: {user_query}"

    payload = {"contents": [{"parts": [{"text": prompt}]}]}

    try:
        response = requests.post(
            f"{config.LLM_API_URL}?key={config.GEMINI_API_KEY}",
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload)
        )
        if response.status_code == 200:
            result = response.json()
            text = result.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
            return text.strip()
    except Exception:
        pass

    return user_query


def query_qdrant_rag(user_query: str, chat_history: list, page_filter: int = None):

    dense_model = config.get_dense_model()
    sparse_model = config.get_sparse_model()
    client = config.get_qdrant_client()

    if not dense_model or not sparse_model or not client:
        return "LLM Error (Missing Resources)", []

    try:
        if not client.collection_exists(collection_name=config.COLLECTION_NAME):
            return "Collection not found. Please ingest a document.", []
    except Exception as e:
        return f"Connection Error: {e}", []

    # Query refinement already done by graph
    refined_query = user_query  

    # ----- Hybrid Search -----
    dense_query = dense_model.embed_query(refined_query)

    sparse_embedding_obj = list(sparse_model.embed([refined_query]))[0]
    sparse_query = models.SparseVector(
        indices=sparse_embedding_obj.indices.tolist(),
        values=sparse_embedding_obj.values.tolist()
    )

    prefetch = [
        models.Prefetch(
            query=dense_query,
            using=config.DENSE_VECTOR_NAME,
            limit=20,
        ),
        models.Prefetch(
            query=sparse_query,
            using=config.SPARSE_VECTOR_NAME,
            limit=20,
        ),
    ]

    hybrid_result = client.query_points(
        collection_name=config.COLLECTION_NAME,
        prefetch=prefetch,
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=5,
        with_payload=True
    )

    if not hybrid_result.points:
        return "No matching content found.", []

    retrieved_docs = []
    context_parts = []

    for point in hybrid_result.points:
        payload = point.payload
        content = payload.get("page_content", "")
        page = payload.get("page_number", "?")

        context_str = f"[Page {page}]: {content}"
        context_parts.append(context_str)

        retrieved_docs.append({
            "content": content,
            "page": page,
            "score": point.score
        })

    full_context = "\n\n".join(context_parts)

    final_prompt = f"CONTEXT:\n{full_context}\n\nUSER QUESTION: {refined_query}"

    payload = {
        "contents": [{"parts": [{"text": final_prompt}]}],
        "systemInstruction": {"parts": [{"text": config.RAG_SYSTEM_PROMPT}]},
    }

    response = requests.post(
        f"{config.LLM_API_URL}?key={config.GEMINI_API_KEY}",
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload)
    )

    result = response.json()
    answer = result.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")

    return answer, retrieved_docs


# ---------------------------------------
# LangGraph Wrapper (FIXED)
# ---------------------------------------

from rag_graph import rag_graph

def run_rag_with_graph(user_query: str, chat_history: list):
    # Initial empty state
    state = {
        "user_query": user_query,
        "refined_query": None,
        "answer": None,
        "retrieved_docs": None,
        "chat_history": chat_history,
        "timings": {}
    }

    result_state = rag_graph.invoke(state)

    return (
        result_state.get("answer"),
        result_state.get("retrieved_docs"),
        result_state.get("timings"),
    )
