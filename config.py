import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load environment variables
load_dotenv()

# ---------------- API KEYS ----------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# ---------------- QDRANT CONFIG ----------------
COLLECTION_NAME = "pdf_rag_collection"
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
QDRANT_URL = f"http://{QDRANT_HOST}:{QDRANT_PORT}"
VECTOR_SIZE = 384  # for all-MiniLM-L6-v2

# ---------------- LLM CONFIG ----------------
LLM_MODEL = "gemini-2.5-flash"
LLM_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{LLM_MODEL}:generateContent"

# ---------------- EMBEDDINGS MODEL ----------------
# Singleton pattern to avoid reloading the model on every import
_EMBEDDINGS_MODEL = None

def get_embeddings_model():
    """Return the initialized embeddings model."""
    global _EMBEDDINGS_MODEL
    if _EMBEDDINGS_MODEL is None:
        try:
            # Using all-MiniLM-L6-v2 for efficient, lightweight embeddings
            _EMBEDDINGS_MODEL = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        except Exception as e:
            st.error(f"Error initializing HuggingFaceEmbeddings: {e}. Please ensure 'sentence-transformers' is installed.")
            return None
    return _EMBEDDINGS_MODEL