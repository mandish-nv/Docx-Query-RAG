import streamlit as st
from utils.auth import Authentication

ROLE_PAGES = {
    "user": [
        ("pages/Legal_Assistant.py", "📜 Legal Assistant"),
    ],
    "admin": [
        ("pages/Legal_Assistant.py", "📜 Legal Assistant"),
        ("pages/Organization_Assistant.py", "🏢 Organization Assistant"),
        ("pages/Document_Ingestion.py", "📄 Document Ingestion"),
        ("pages/Rule_Generator.py", "⚙️ Rule Generator"),
    ],
    "developer": [
        ("pages/Legal_Assistant.py", "📜 Legal Assistant"),
        ("pages/Organization_Assistant.py", "🏢 Organization Assistant"),
        ("pages/Document_Ingestion.py", "📄 Document Ingestion"),
        ("pages/Rule_Generator.py", "⚙️ Rule Generator"),
    ],
    "employee": [
        ("pages/Legal_Assistant.py", "📜 Legal Assistant"),
        ("pages/Organization_Assistant.py", "🏢 Organization Assistant"),
    ],
}


def init_page(title):
    # MUST be first Streamlit call
    st.set_page_config(
        page_title=f"⚖️ Kanun Mitra - {title}",
        layout="wide"
    )

    # ---- Hide Streamlit default navigation ----
    st.markdown(
        """
        <style>
            [data-testid="stSidebarNav"] {
                display: none;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    auth = Authentication()
    if not auth.check_session():
        st.warning("Please login to access this page.")
        st.stop()

    user = auth.get_current_user()
    role = user["role"].lower()

    allowed_pages = ROLE_PAGES.get(role, [])

    # ---- Custom Sidebar ----
    with st.sidebar:
        st.title("⚖️ Kanun Mitra")

        st.markdown(f"**User:** {user['username']}")
        st.markdown(f"**Role:** `{role.upper()}`")

        if st.button("Logout", type="secondary", use_container_width=True):
            auth.logout()
            st.rerun()

        st.divider()
        st.caption("Navigation")

        for page_path, label in allowed_pages:
            st.page_link(page_path, label=label)

    return user
