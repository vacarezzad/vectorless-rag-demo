import sys
import os

# Ensure core and sections are importable regardless of working directory
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st

from sections import educational, code_showcase, demo

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Vectorless RAG — Demo",
    page_icon="🌲",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
        .stTabs [data-baseweb="tab"] { font-size: 1rem; padding: 0.5rem 1.2rem; }
        .stDataFrame { font-size: 0.9rem; }
        pre { font-size: 0.85rem !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("🌲 Vectorless RAG")
st.caption(
    "Retrieval-Augmented Generation sin vectores, sin embeddings, sin vector database."
)

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_edu, tab_code, tab_demo = st.tabs(
    ["📚 ¿Qué es?", "💻 Implementación", "🔬 Demo Práctica"]
)

with tab_edu:
    educational.render()

with tab_code:
    code_showcase.render()

with tab_demo:
    demo.render()
