import streamlit as st
import tempfile
import os
from ingestion_pipeline import ingest_documents_to_qdrant
from utils.ui_components import init_page

user_info = init_page("Document Ingestion")

# st.set_page_config(page_title="Document Ingestion")

st.markdown("""
    <style>
        [data-testid="stSidebarNav"] ul li:first-child {
            display: none;
        }
    </style>
""", unsafe_allow_html=True)

st.title("📄 Document Ingestion")
st.write("Upload PDF laws or company documents to add them to the AI knowledge base.")

# with st.sidebar:
#     if st.button("⬅ Back to Dashboard"):
#         st.switch_page("main.py") # or your main dashboard file

uploaded_files = st.file_uploader("Choose a PDF file", type=["pdf"],accept_multiple_files=True)
if uploaded_files: 
    if st.button("Process Documents"):
        with st.spinner("Ingesting documents..."):
            # Pass the user role to the ingestion function
            user_role = user_info.get("role", "user").lower()
            
            # Note: Your ingestion_pipeline expects a list of file objects or paths
            # In your current script, you are looping and sending paths. 
            # I will keep your loop logic but pass the role.
            
            for uploaded_file in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name
                
                try:
                    # Added user_role argument
                    ingest_documents_to_qdrant(tmp_path, user_role=user_role)
                finally:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
            
            st.success(f"Processed {len(uploaded_files)} files!")
            
# if uploaded_file:
    
#     if st.button("Start Ingestion", type="primary"):
#         with st.spinner("Processing PDF and creating embeddings..."):
#             with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
#                 tmp.write(uploaded_file.getvalue())
#                 tmp_path = tmp.name
#             try:
#                 ingest_documents_to_qdrant(tmp_path)
#                 st.success("Success! The knowledge base has been updated.")
#             except Exception as e:
#                 st.error(f"Error: {e}")
#             finally:
#                 if os.path.exists(tmp_path):
#                     os.remove(tmp_path)

