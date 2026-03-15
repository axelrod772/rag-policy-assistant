"""Streamlit UI for RAG Policy Assistant. Point to API (default http://127.0.0.1:8000)."""
import streamlit as st
import requests

st.set_page_config(page_title="RAG Q&A", page_icon="🔎", layout="wide")
st.title("RAG Policy Assistant (Hybrid Search)")

api_base = st.text_input("API base URL", value="http://127.0.0.1:8000")
query = st.text_input("Ask a question", placeholder="e.g., What is the waiting period for pre-existing diseases?")
alpha = st.slider("Dense weight (alpha)", 0.0, 1.0, 0.5, 0.05)
top_k = st.slider("Top K", 1, 10, 5, 1)
max_toks = st.slider("Max new tokens", 32, 512, 160, 32)
temp = st.slider("Temperature", 0.0, 1.0, 0.2, 0.1)
use_reranker = st.checkbox("Use reranker (CrossEncoder)", value=True)
pool = st.slider("Candidate pool (for reranker)", 5, 50, 20, 1)

if st.button("Ask") and len(query) >= 3:
    with st.spinner("Thinking..."):
        try:
            r = requests.post(
                f"{api_base}/query",
                json={
                    "query": query,
                    "top_k": top_k,
                    "alpha": alpha,
                    "max_new_tokens": max_toks,
                    "temperature": temp,
                    "use_reranker": use_reranker,
                    "candidates": pool,
                },
                timeout=180,
            )
            if r.ok:
                data = r.json()
                st.subheader("Answer")
                st.write(data["answer"])
                st.caption(f"Latency: {data.get('latency_ms', 0)} ms")
                st.subheader("Sources")
                for i, doc in enumerate(data.get("source_documents", [])):
                    meta = doc.get("metadata", {})
                    st.markdown(f"[{i+1}] {meta.get('source', '—')} — page {meta.get('page', '—')}")
                    st.caption(doc.get("content", "")[:500])
            else:
                st.error(f"Error {r.status_code}: {r.text}")
        except Exception as e:
            st.error(str(e))

st.caption("Tip: alpha=0.5 gives 50/50 hybrid (FAISS + BM25).")
