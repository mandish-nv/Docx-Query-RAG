import streamlit as st
import os
import tempfile

from ingestion_pipeline import ingest_documents_to_qdrant
from rag_graph import run_rag_with_graph 
from rag_query import generate_compliant_rules 

st.set_page_config(page_title="Docx-Query-RAG", layout="wide")

def main():
    st.title("Docx-Query-RAG")

    # --- Sidebar Navigation & Config ---
    with st.sidebar:
        st.header("Navigation")
        app_mode = st.radio("Choose Mode", ["Chat with PDF", "Rule Generation"])
        
        st.divider()
        st.header("Document Ingestion")
        uploaded_file = st.file_uploader("Upload PDF (Laws/Docs)", type=["pdf"])

        if uploaded_file:
            if st.button("Process Document"):
                with st.spinner("Ingesting (Hybrid Mode)..."):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(uploaded_file.getvalue())
                        tmp_path = tmp.name
                    try:
                        ingest_documents_to_qdrant(tmp_path)
                    finally:
                        os.remove(tmp_path)

    # --- MODE: CHAT WITH PDF ---
    if app_mode == "Chat with PDF":
        st.subheader("Interactive Knowledge Base")
        st.caption("Ask questions about the uploaded documents.")

        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Show chat history
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if user_query := st.chat_input("Ask something from the PDF..."):
            st.session_state.messages.append({"role": "user", "content": user_query})
            with st.chat_message("user"):
                st.markdown(user_query)

            with st.chat_message("assistant"):
                with st.spinner("Processing..."):
                    answer, docs, timings, refined_queries = run_rag_with_graph(
                        user_query, 
                        st.session_state.messages[:-1]
                    )

                    st.markdown(answer)

                    # Transparency Section
                    with st.expander("Details: Query Refinement & Timing"):
                        st.subheader("Generated Queries")
                        if refined_queries:
                            for q in refined_queries:
                                st.text(f"- {q}")
                        st.subheader("Timing Report (ms)")
                        st.json(timings)

                    # Retrieved Context
                    if docs:
                        with st.expander("Retrieved Context (Top Re-ranked)"):
                            for i, d in enumerate(docs):
                                st.markdown(f"**{i+1}. Page {d['page']} — Re-rank Score: {d['score']:.4f}**")
                                st.info(d["content"])

            st.session_state.messages.append({"role": "assistant", "content": answer})

    # --- MODE: RULE GENERATION ---
    elif app_mode == "Rule Generation":
        st.subheader("Intelligent Rule Generator & Compliance Auditor")
        st.caption("Generate organizational rules backed by your document database (National/Local Laws).")

        # Sidebar inputs for Rule Generation specifics
        with st.sidebar:
            st.divider()
            st.markdown("### Rule Parameters")
            rule_context = st.text_input("Organization/Context Type", value="IT Organization", help="E.g., IT Company, Factory, Hospital")
            custom_rules_input = st.text_area("Custom Rules / Desires", height=150, placeholder="E.g., We want flexible working hours, remote work options, but strict data security.")
            generate_btn = st.button("Generate & Check Compliance")

        if generate_btn:
            if not rule_context or not custom_rules_input:
                st.warning("Please fill in both the Organization Type and Custom Rules.")
            else:
                with st.spinner("Retrieving laws, drafting rules, and checking compliance..."):
                    # Call the new function in rag_query.py
                    generated_rules, compliance_report, source_docs = generate_compliant_rules(
                        rule_context, 
                        custom_rules_input
                    )

                # Layout: 2 Columns (Rules vs Compliance)
                col1, col2 = st.columns([1, 1])

                with col1:
                    st.success("Drafted Rules")
                    st.markdown(generated_rules)

                with col2:
                    st.error("Compliance Audit Report")
                    st.markdown(compliance_report)
                
                # Show Sources Used
                st.divider()
                with st.expander("View Referenced Legal Context (Source Data)"):
                    if source_docs:
                        for i, d in enumerate(source_docs):
                            st.markdown(f"**Source {i+1} (Page {d['page']})**")
                            st.info(d["content"])
                    else:
                        st.write("No specific laws were found to reference.")

if __name__ == "__main__":
    main()
