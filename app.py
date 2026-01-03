import streamlit as st
import sys
import os
import tempfile
import time
import pandas as pd

# Import logic from old code
from ingestion_pipeline import ingest_documents_to_qdrant
from rag_graph import run_rag_with_graph 
from rag_query import generate_compliant_rules 
import config
from dashboard import show_dashboard

# Import Auth from new code
from utils.auth import Authentication

# Page configuration
st.set_page_config(
    page_title="KanunMitra - Legal Assistance Portal",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded" # Changed to expanded so it's visible after login
)

# Initialize authentication
auth = Authentication()
is_logged_in = auth.check_session()

# --- DYNAMIC CSS STYLES ---
if not is_logged_in:
    # Hide sidebar entirely on login page
    sidebar_style = """
    <style>
        [data-testid="stSidebar"] {
            display: none;
        }
    </style>
    """
else:
    # Hide ONLY the "app" (first) link in the sidebar navigation
    sidebar_style = """
    <style>
        /* Target the navigation list and hide the first item (usually the main script) */
        [data-testid="stSidebarNav"] ul li:first-child {
            display: none;
        }
        
        /* Optional: Remove the extra padding/gap left by the hidden item */
        [data-testid="stSidebarNav"] {
            padding-top: 0rem;
        }
    </style>
    """
    sidebar_style = """
    """

st.markdown(sidebar_style, unsafe_allow_html=True)

st.markdown("""
<style>
    .stApp { overflow: hidden !important; max-height: 100vh !important; }
    ::-webkit-scrollbar { display: none !important; }
    * { scrollbar-width: none !important; }
    .main { overflow: hidden !important; height: 100vh !important; padding: 0; margin: 0; }
    .block-container { padding-top: 1rem !important; padding-bottom: 1rem !important; }
    
    /* Login Button Styling */
    form[data-testid="stForm"] div[data-testid="column"] button[kind="formSubmit"] {
        background-color: #0d6efd !important;
        color: white !important;
        padding: 1.5rem 4rem !important;
        font-size: 1.4rem !important;
        font-weight: bold !important;
        border-radius: 12px !important;
        width: 100% !important;
        height: 70px !important;
        margin-top: 30px !important;
    }
    
    .role-badge {
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
        color: white;
    }
    .admin { background-color: #ff4b4b; }
    .user { background-color: #0083B8; }
</style>
""", unsafe_allow_html=True)

def show_login_form():
    left_col, right_col = st.columns([2, 1])
    with right_col:
        st.markdown('<h1 style="text-align: center; color: #ffffff;">KanunMitra</h1>', unsafe_allow_html=True)
        st.markdown('<h3 style="text-align: center; color: #ffffff;">Legal Assistance Portal</h3>', unsafe_allow_html=True)
        
        with st.form("login_form"):
            username = st.text_input("**Username**", placeholder="Enter your username", key="login_username")
            password = st.text_input("**Password**", type="password", placeholder="Enter your password", key="login_password")
            
            st.markdown("<br>", unsafe_allow_html=True)
            btn_col = st.columns([1, 2, 1])[1]
            with btn_col:
                login_button = st.form_submit_button("**LOGIN**", use_container_width=True, type="primary")
            
            if login_button:
                if not username or not password:
                    st.error("Please fill in all fields!")
                else:
                    user = auth.authenticate(username, password)
                    if user:
                        st.session_state['user'] = user
                        st.success(f"Welcome, {user['username']}!")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("Invalid credentials!")

    with left_col:
        try:
            st.image("images/RobotLaw.png", width=600)
        except:
            st.info("Legal AI System Ready")

def show_main_dashboard():
    st.title("⚖️ KanunMitra Dashboard")

    # --- Sidebar Content (Authenticated Only) ---
    with st.sidebar:
        user = auth.get_current_user()
        st.markdown(f"### User Profile")
        st.markdown(f"**Logged in as:** {user['username']}")
        st.markdown(f"<span class='role-badge user'>{user['role'].upper()}</span>", unsafe_allow_html=True)
        
        if st.button("Logout", type="secondary", use_container_width=True):
            auth.logout()
            st.rerun()
            
        st.divider()
    
    # show_dashboard()

def main():
    if is_logged_in:
        show_main_dashboard()
    else:
        show_login_form()

if __name__ == "__main__":
    main()