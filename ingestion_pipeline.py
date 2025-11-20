import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient, models
import config

def ingest_documents_to_qdrant(pdf_path):
    """Loads PDF, chunks it, creates Qdrant collection, and stores vectors."""
    
    embeddings_model = config.get_embeddings_model()

    if not embeddings_model:
        st.error("Embeddings model not loaded. Ingestion cannot proceed.")
        return

    st.info("Starting document ingestion...")

    # 1. Load PDF
    try:
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        st.success(f"Loaded {len(documents)} page(s).")
    except Exception as e:
        st.error(f"Error loading PDF: {e}")
        return

    # 2. Chunking
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len
    )
    chunks = splitter.split_documents(documents)
    st.success(f"Split into {len(chunks)} chunks.")

    # 3. Qdrant ingestion
    try:
        # Initialize Client Explicitly
        client = QdrantClient(url=config.QDRANT_URL)

        # Check and recreate collection
        if client.collection_exists(collection_name=config.COLLECTION_NAME):
            client.delete_collection(collection_name=config.COLLECTION_NAME)
            st.info(f"Deleted existing collection '{config.COLLECTION_NAME}'.")
        
        client.create_collection(
            collection_name=config.COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=config.VECTOR_SIZE,
                distance=models.Distance.COSINE,
            ),
        )
        st.success("Created new collection structure.")

        vector_store = Qdrant(
            client=client,
            collection_name=config.COLLECTION_NAME,
            embeddings=embeddings_model
        )
        
        vector_store.add_documents(documents=chunks)

        st.balloons()
        info = client.get_collection(collection_name=config.COLLECTION_NAME)
        st.success(f"Stored {len(chunks)} vectors in Qdrant. Total vectors: {info.points_count}")

    except Exception as e:
        st.error(f"Qdrant ingestion error: {e}")