import os
import pickle
import streamlit as st
from typing import List, Optional

from helpers.retriever import hybrid_search_with_rerank
from helpers.chain import build_rag_chain, build_summary_chain
from langchain_groq import ChatGroq
from langchain.schema import Document
from helpers.vectorstore import get_vectorstore, get_bm25

# Caching loaders
@st.cache_resource(show_spinner=False)
def load_vectorstore(faiss_dir: str):
    return get_vectorstore(faiss_dir=faiss_dir)

@st.cache_resource(show_spinner=False)
def load_bm25(bm25_file: str):
    return get_bm25(bm25_file=bm25_file)

@st.cache_resource(show_spinner=False)
def load_reranker():
    from helpers.retriever import load_reranker
    return load_reranker()

@st.cache_resource(show_spinner=False)
def load_categories_from_chunks(chunks_file: str) -> List[str]:
    try:
        with open(chunks_file, "rb") as f:
            chunks = pickle.load(f)
        return sorted({d.metadata.get("category", "uncategorized") for d in chunks if d.metadata.get("category")})
    except Exception as e:
        st.error(f"Failed to load categories: {e}")
        return ["all"]

@st.cache_resource(show_spinner=False)
def get_llm(model_name: str, temperature: float):
    if not os.getenv("GROQ_API_KEY"):
        st.warning("GROQ_API_KEY not set in environment. Set it before running.")
    return ChatGroq(model=model_name, temperature=temperature)

# UI
st.set_page_config(page_title="EFDA RAG Assistant", page_icon="💊", layout="wide")

st.title("💊 EFDA RAG Assistant (Hybrid FAISS + BM25 with Reranking)")
st.caption("Streamlit UI with Reranked Hybrid Retrieval + Sources, EFDA(Ethiopian Food and Drug Administration)")

with st.sidebar:
    st.subheader("Settings")
    chunk_files = [f for f in os.listdir() if f.endswith(".pkl") and f.startswith("chunks_")]
    chunk_file = st.selectbox("Select document set", options=["chunks.pkl"] + chunk_files if chunk_files else ["chunks.pkl"])
    model = st.text_input("Groq model", value="llama-3.3-70b-versatile")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.1, 0.05)
    k = st.slider("Top-k results", 3, 12, 6, 1)
    rerank_top_n = st.slider("Rerank Top N", 10, 50, 20, 5)
    categories = load_categories_from_chunks(chunk_file)
    category = st.selectbox("Filter by category", options=["all"] + [c for c in categories if c != "all"])
    st.markdown("---")
    st.write("`GROQ_API_KEY` must be set in your environment before running.")

col1, col2 = st.columns([2, 1])

with col1:
    question = st.text_area("Ask a question about EFDA guidelines:", height=120, placeholder="e.g., What is the process for variation applications to registered medicines?")
    col1a, col1b = st.columns(2)
    with col1a:
        run_qa = st.button("Search & Answer", type="primary", use_container_width=True)
    with col1b:
        run_summary = st.button("Summarize Documents", type="secondary", use_container_width=True)

    if run_qa and question.strip():
        filters = {"category": category} if category and category.lower() != "all" else None
        with st.spinner("Retrieving..."):
            faiss_dir = f"{chunk_file.replace('.pkl', '')}_faiss_store"
            bm25_file = f"{chunk_file.replace('.pkl', '')}_bm25.pkl"
            docs = hybrid_search_with_rerank(question.strip(), k=k, filters=filters, rerank_top_n=rerank_top_n)
        if not docs:
            st.error("No relevant documents found.")
        else:
            with st.spinner("Generating answer..."):
                llm = get_llm(model, temperature)
                chain = build_rag_chain(llm)
                ans = chain.invoke({"question": question.strip(), "docs": docs}).content
            st.markdown("### Answer")
            st.write(ans)

    if run_summary:
        filters = {"category": category} if category and category.lower() != "all" else None
        with st.spinner("Retrieving..."):
            faiss_dir = f"{chunk_file.replace('.pkl', '')}_faiss_store"
            bm25_file = f"{chunk_file.replace('.pkl', '')}_bm25.pkl"
            docs = hybrid_search_with_rerank(question.strip() if question else "Summarize EFDA guidelines", k=k, filters=filters, rerank_top_n=rerank_top_n)
        if not docs:
            st.error("No relevant documents found.")
        else:
            with st.spinner("Generating summary..."):
                llm = get_llm(model, temperature)
                chain = build_summary_chain(llm)
                summary = chain.invoke({"docs": docs}).content
            st.markdown("### Summary")
            st.write(summary)

with col2:
    st.markdown("### Retrieved Chunks")
    st.caption("Top chunks used as context (title · page · preview).")
    if "docs" in locals() and docs:
        for i, d in enumerate(docs, 1):
            title = d.metadata.get("doc_title", "unknown")
            page = d.metadata.get("page", "?")
            with st.expander(f"{i}. {title} · p.{page}"):
                st.write(d.page_content)
                st.caption(f"Category: {d.metadata.get('category', 'N/A')}")
