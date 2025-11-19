import streamlit as st
import os
import tempfile

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse

## --- Configuration ---
COLLECTION_NAME = "pdf_rag_collection"
# Setting these to connect to a running Qdrant instance
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
# The vector size for 'sentence-transformers/all-MiniLM-L6-v2' is 384
VECTOR_SIZE = 384 

# 1. & 2. Embedding Model (Dense)
# Use the HuggingFaceEmbeddings wrapper for sentence-transformers/all-MiniLM-L6-v2
@st.cache_resource
def get_dense_embedding_model():
    """Loads and caches the Hugging Face Sentence Transformer model."""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

## --- Functions ---

def load_and_process_pdf(pdf_path, embeddings_model):
    """
    1. Loads PDF, 2. Chunks text, and 3. Stores embeddings in Qdrant.
    """
    st.info("Starting document ingestion and processing...")

    # 1. Document Ingestion
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    st.success(f"Loaded {len(documents)} page(s) from PDF.")

    # 2. Chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_documents(documents)
    st.success(f"Chunked document into {len(chunks)} text snippets.")

    # 3. Vector Store: Store embeddings in Qdrant
    try:
        # Initialize the Qdrant Client (Connecting to persistent instance)
        # To run in memory, change: client = QdrantClient(":memory:")
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

        # --- NEW LOGIC: Check and Create Collection ---
        try:
            # Check if the collection exists
            client.get_collection(collection_name=COLLECTION_NAME)
            st.info(f"Collection **{COLLECTION_NAME}** found. Appending vectors...")
        except UnexpectedResponse:
            # If not found, create it
            st.warning(f"Collection **{COLLECTION_NAME}** not found. Creating it now...")
            client.recreate_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=models.VectorParams(size=VECTOR_SIZE, distance=models.Distance.COSINE),
            )
            st.success(f"Collection **{COLLECTION_NAME}** created successfully.")
        # --- END NEW LOGIC ---
        
        # Store embeddings - We fix the issue by:
        # 1. Initializing the LangChain Qdrant object with the existing client.
        # 2. Calling add_documents() to upload the chunks.
        
        # 1. Initialize LangChain Qdrant object
        vector_store = Qdrant(
            client=client,
            collection_name=COLLECTION_NAME,
            embeddings=embeddings_model # The Qdrant constructor expects 'embeddings' not 'embedding'
        )

        # 2. Add documents (embed and upload)
        vector_store.add_documents(
            documents=chunks,
        )

        st.balloons()
        st.success(f"Successfully stored {len(chunks)} vectors in Qdrant collection: **{COLLECTION_NAME}**")
        
        # Example check: Get the total collection point count
        collection_info = client.get_collection(collection_name=COLLECTION_NAME)
        st.write(f"Total Vector Count in Collection: {collection_info.points_count}")
        
    except Exception as e:
        st.error(f"An error occurred during Qdrant connection or storage: {e}")


## --- Streamlit UI ---

st.title("📚 PDF Ingestion to Qdrant Vector Store")
st.markdown("Uses `sentence-transformers/all-MiniLM-L6-v2` and LangChain/Qdrant.")


uploaded_file = st.file_uploader(
    "**1. Upload a PDF Document**",
    type="pdf"
)

# Load the model once
embedding_model = get_dense_embedding_model()
st.sidebar.markdown(f"**Loaded Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2`")
st.sidebar.markdown(f"**Chunk Size:** 500 tokens")

if uploaded_file is not None:
    # Use a temporary file to save the uploaded PDF for PyPDFLoader
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name
        
    st.subheader(f"Processing File: **{uploaded_file.name}**")
    
    if st.button("🚀 Ingest & Embed"):
        load_and_process_pdf(tmp_file_path, embedding_model)
        
    # Clean up the temporary file
    os.remove(tmp_file_path)

else:
    st.info("Please upload a PDF file to begin the ingestion process.")