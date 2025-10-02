"""Streamlit interface for the RAG chatbot."""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from generation.dummy import EchoGenerator
from indexing.embedder import get_default_embedder
from pipelines.chatbot import RAGPipeline

st.set_page_config(page_title="RAG Chatbot", layout="wide")

DATA_PATH = st.sidebar.text_input("Data path", value="data/sample_docs")
TOP_K = st.sidebar.slider("Top K", min_value=1, max_value=10, value=5)

if "pipeline" not in st.session_state or st.session_state.get("data_path") != DATA_PATH:
    st.session_state["pipeline"] = RAGPipeline(
        data_path=Path(DATA_PATH),
        embedder=get_default_embedder(prefer_lightweight=True),
        generator=EchoGenerator(),
        use_faiss=False,
    )
    st.session_state["data_path"] = DATA_PATH

st.title("RAG Chatbot Demo")
query = st.text_input("Ask a question")

if st.button("Submit") and query:
    docs = st.session_state["pipeline"].query(query, top_k=TOP_K)
    response = st.session_state["pipeline"].chat(query, top_k=TOP_K)
    st.subheader("Response")
    st.write(response)
    st.subheader("Retrieved Documents")
    for doc in docs:
        with st.expander(doc.doc_id):
            st.write(doc.content)
