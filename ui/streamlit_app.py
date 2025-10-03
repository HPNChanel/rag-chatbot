"""Streamlit interface for the RAG chatbot."""
from __future__ import annotations

import os
import re
from typing import Any, Dict

import requests
import streamlit as st

st.set_page_config(page_title="RAG Chatbot", layout="wide")


def highlight_matches(text: str, query: str) -> str:
    """Return HTML with occurrences of query terms highlighted."""

    if not text or not query:
        return text
    terms = {term.lower() for term in re.split(r"\W+", query) if term}
    if not terms:
        return text

    def _replacer(match: re.Match[str]) -> str:
        word = match.group(0)
        if word.lower() in terms:
            return f"<mark>{word}</mark>"
        return word

    pattern = re.compile(r"\b(" + "|".join(map(re.escape, terms)) + r")\b", flags=re.IGNORECASE)
    return pattern.sub(_replacer, text)


def _post(base_url: str, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    response = requests.post(f"{base_url}{endpoint}", json=payload, timeout=30)
    response.raise_for_status()
    return response.json()


API_BASE_URL = st.sidebar.text_input(
    "API base URL",
    value=os.environ.get("RAG_API_URL", "http://localhost:8000"),
)
TOP_K = st.sidebar.slider("Top K", min_value=1, max_value=10, value=4)

if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "last_result" not in st.session_state:
    st.session_state["last_result"] = None

st.title("RAG Chatbot Demo")

prompt = st.chat_input("Ask the chatbot about the sample docs")
if prompt:
    st.session_state["messages"].append({"role": "user", "content": prompt})
    try:
        with st.spinner("Generating answer..."):
            payload = _post(
                API_BASE_URL,
                "/chat",
                {
                    "query": prompt,
                    "top_k": TOP_K,
                },
            )
    except requests.RequestException as exc:
        st.error(f"Failed to contact API: {exc}")
        payload = None
    if payload:
        answer = payload.get("answer", "")
        st.session_state["messages"].append({"role": "assistant", "content": answer})
        st.session_state["last_result"] = payload

left, right = st.columns((2, 1))

with left:
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if st.session_state["last_result"]:
        docs = st.session_state["last_result"].get("documents", [])
        query_text = st.session_state["last_result"].get("query", "")
        st.subheader("Retrieved Documents")
        for doc in docs:
            metadata = doc.get("metadata", {})
            label = metadata.get("title") or doc.get("doc_id", "Document")
            scores = doc.get("scores", {})
            header = f"{label} — rerank: {scores.get('reranked', 0):.3f}" if scores else label
            with st.expander(header):
                highlighted = highlight_matches(doc.get("content", ""), query_text)
                st.markdown(highlighted, unsafe_allow_html=True)

with right:
    st.subheader("Metrics")
    metrics = (st.session_state["last_result"] or {}).get("metrics")
    if metrics:
        latency = metrics.get("end_to_end_time", 0.0)
        docs_count = len(st.session_state["last_result"].get("documents", []))
        reranker = metrics.get("reranker_impact", 0.0)
        st.metric("Latency", f"{latency*1000:.1f} ms")
        st.metric("Retrieved", str(docs_count))
        st.metric("Reranker Δ", f"{reranker:.2f}")
        st.caption(
            "Retrieval: {retrieval:.1f} ms • Generation: {generation:.1f} ms".format(
                retrieval=metrics.get("retrieval_time", 0.0) * 1000,
                generation=metrics.get("generation_time", 0.0) * 1000,
            )
        )
    else:
        st.info("Run a query to see metrics.")

    if st.session_state["last_result"]:
        st.subheader("Citations")
        for citation in st.session_state["last_result"].get("citations", []):
            title = citation.get("title") or citation.get("doc_id")
            source = citation.get("source")
            excerpt = citation.get("excerpt", "")
            st.markdown(f"**{title}**")
            if source:
                st.caption(source)
            st.write(excerpt)
