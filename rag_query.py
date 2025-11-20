import streamlit as st
import requests
import json
import time
from qdrant_client import QdrantClient
from langchain_core.documents import Document
import config

def query_qdrant_rag(user_query: str):
    """Performs similarity search in Qdrant, then uses the retrieved context for RAG with Gemini."""
    
    embeddings_model = config.get_embeddings_model()

    if not embeddings_model:
        return "LLM Error (Missing Embeddings Model)", []

    st.info(f"Searching Qdrant for: {user_query}")

    try:
        # Initialize Client Explicitly
        client = QdrantClient(url=config.QDRANT_URL)

        if not client.collection_exists(collection_name=config.COLLECTION_NAME):
            st.warning("Collection does not exist. Please ingest a document first.")
            return "No documents ingested yet.", []

    except Exception as e:
        st.error(f"Error connecting to Qdrant: {e}")
        return f"Error connecting to Qdrant: {e}", []

    # Retrieve top-k documents using query_points
    try:
        # 1. Convert query text to vector manually
        query_vector = embeddings_model.embed_query(user_query)

        # 2. Use client.query_points instead of vector_store.similarity_search
        search_result = client.query_points(
            collection_name=config.COLLECTION_NAME,
            query=query_vector,
            limit=4,
            with_payload=True
        )

        if not search_result.points:
            st.warning("No relevant documents found.")
            return "No matching content in the PDF.", []

        # 3. Map Qdrant points back to LangChain Document format
        retrieved_docs = []
        for point in search_result.points:
            content = point.payload.get("page_content", "")
            metadata = point.payload.get("metadata", {})
            retrieved_docs.append(Document(page_content=content, metadata=metadata))

        # This logic joins the retrieved data into a context string
        context = "\n---\n".join(doc.page_content for doc in retrieved_docs)
        st.success("Retrieved relevant document chunks.")

    except Exception as e:
        st.error(f"Similarity search error: {e}")
        return f"Search error: {e}", []

    # LLM Query
    system_instruction = (
        "You are an expert Q&A model. Only answer using the provided CONTEXT. "
        "If the answer is not in the context, say so clearly."
    )

    prompt = f"CONTEXT:\n{context}\n\nQUESTION: {user_query}"

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "systemInstruction": {"parts": [{"text": system_instruction}]},
    }
    llm_api_endpoint = f"{config.LLM_API_URL}?key={config.GEMINI_API_KEY}"

    for attempt in range(3):
        try:
            response = requests.post(
                llm_api_endpoint,
                headers={"Content-Type": "application/json"},
                data=json.dumps(payload)
            )
            response.raise_for_status()

            result = response.json()
            candidate = result.get("candidates", [{}])[0]
            answer = candidate.get("content", {}).get("parts", [{}])[0].get("text")

            if answer:
                return answer, retrieved_docs
            else:
                st.warning("LLM returned an empty or malformed response.")
                if attempt < 2:
                    time.sleep(1.5 ** attempt)
                else:
                    st.error("LLM API failed after retries.")
                    return "LLM Error (Empty Response)", []

        except requests.exceptions.RequestException as e:
            if attempt < 2:
                time.sleep(1.5 ** attempt)
            else:
                st.error(f"LLM API failed after retries. Request Error: {e}")
                return "LLM Error (Request Failure)", []
        except Exception as e:
            if attempt < 2:
                time.sleep(1.5 ** attempt)
            else:
                st.error(f"An unexpected error occurred: {e}")
                return "LLM Error (Unexpected Exception)", []

    return "Failed to generate response after all attempts.", []