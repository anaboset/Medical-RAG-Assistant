import os
import pickle
import streamlit as st
from typing import List, Optional, Dict
import tempfile
from helpers.retriever import hybrid_search_with_rerank
from helpers.chain import build_rag_chain, build_summary_chain
from langchain_groq import ChatGroq
from langchain.schema import Document
from helpers.vectorstore import get_vectorstore, get_bm25, store_chunks, get_bm25_retriever
from helpers.pdfloader import load_pdfs
from helpers.chunker import chunk_documents

# Initialize session state for conversation memory
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []


# Caching loaders with hash functions to invalidate cache on file changes
@st.cache_resource(show_spinner=False, hash_funcs={str: lambda x: os.path.getmtime(x) if os.path.exists(x) else 0})
def load_vectorstore(faiss_dir: str) -> ChatGroq:
    """Load FAISS vectorstore, cached to avoid repeated loading."""
    return get_vectorstore(faiss_dir=faiss_dir)


@st.cache_resource(show_spinner=False, hash_funcs={str: lambda x: os.path.getmtime(x) if os.path.exists(x) else 0})
def load_bm25(bm25_file: str):
    """Load BM25 retriever from file with error handling."""
    bm25 = get_bm25(bm25_file=bm25_file)
    if bm25 is None:
        st.error(f"BM25 index {bm25_file} not found. Run 'python process_pdfs.py' or upload PDFs to generate it.")
    return bm25


@st.cache_resource(show_spinner=False)
def load_reranker():
    """Load cross-encoder reranker with caching to avoid repeated model loading."""
    from helpers.retriever import load_reranker
    return load_reranker()


@st.cache_resource(show_spinner=False, hash_funcs={str: lambda x: os.path.getmtime(x) if os.path.exists(x) else 0})
def load_categories_from_chunks(chunks_file: str) -> List[str]:
    """
    Extract unique document categories from a pickled chunks file.

    Args:
        chunks_file (str): Path to pickled chunk file.

    Returns:
        List[str]: Sorted list of unique categories; returns ["all"] if failed.
    """
    try:
        with open(chunks_file, "rb") as f:
            chunks = pickle.load(f)
        if not isinstance(chunks, list):
            st.error(f"Invalid data format in {chunks_file}: Expected list, got {type(chunks)}. Regenerate with 'python process_pdfs.py --output_prefix {chunks_file.replace('.pkl', '')}'.")
            return ["all"]
        return sorted({d.metadata.get("category", "uncategorized") for d in chunks if hasattr(d, 'metadata') and d.metadata.get("category")})
    except Exception as e:
        st.error(f"Failed to load categories from {chunks_file}: {e}. Regenerate with 'python process_pdfs.py --output_prefix {chunks_file.replace('.pkl', '')}'.")
        return ["all"]


@st.cache_resource(show_spinner=False)
def get_llm(model_name: str, temperature: float) -> ChatGroq:
    """Initialize LLM with Groq API key"""
    if not os.getenv("GROQ_API_KEY"):
        st.warning("GROQ_API_KEY not set in environment. Set it before running.")
    return ChatGroq(model=model_name, temperature=temperature)


# Process uploaded PDFs
def process_uploaded_pdfs(uploaded_files):
    """Process uploaded PDF files and generate new chunks/indexes."""
    if uploaded_files:
        with tempfile.TemporaryDirectory() as tmp_dir:
            for uploaded_file in uploaded_files:
                file_path = os.path.join(tmp_dir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
            import subprocess
            subprocess.run(["python", "process_pdfs.py", "--input_dir", tmp_dir, "--output_prefix", "uploaded_chunks"])
        st.success("PDFs processed. Please clear cache and reload to use new data.")
    else:
        st.warning("No PDFs uploaded. Using preprocessed default data.")


# UI
st.set_page_config(page_title="Medical RAG Assistant (EFDA Guidelines)", page_icon="💊", layout="wide")

st.title("💊 Medical RAG Assistant (EFDA Guidelines)")
st.caption("Query EFDA medical registration, import, and export regulations")
st.markdown("""This application retrieves and answers questions from Ethiopian Food and Drug Authority (EFDA) medical guidelines PDFs using Retrieval-Augmented Generation (RAG) technology. It includes preprocessed data for immediate use and supports optional PDF uploads for custom content.""")


# -------------------------
# Sidebar
# -------------------------
with st.sidebar:
    st.subheader("Settings")

    # Select chunk file / preloaded document set
    chunk_files = [f for f in os.listdir() if f.endswith(".pkl") and (f.startswith("chunks_") or f == "uploaded_chunks.pkl")]
    chunk_file = st.selectbox("Select document set", options=["chunks.pkl"] + chunk_files if chunk_files else ["chunks.pkl"])

    uploaded_files = st.file_uploader("Upload medical PDF documents (optional)", type="pdf", accept_multiple_files=True)
    if uploaded_files:
        if st.button("Process Uploaded PDFs"):
            process_uploaded_pdfs(uploaded_files)
    st.write("Note: Uploads are optional. The app uses preprocessed EFDA data by default.")

    model = st.text_input("Groq model", value="llama-3.3-70b-versatile")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.1, 0.05)
    k = st.slider("Top-k results", 3, 12, 6, 1)
    rerank_top_n = st.slider("Rerank Top N", 10, 50, 20, 5)

    categories = load_categories_from_chunks(chunk_file)
    category = st.selectbox("Filter by medical category", options=["all"] + [c for c in categories if c != "all"])

    if st.button("Clear Cache (after adding new PDFs or reprocessing)"):
        st.cache_resource.clear()
        st.session_state.conversation_history = []  # Reset memory
        st.success("Cache cleared!")

    st.markdown("---")
    st.write("`GROQ_API_KEY` must be set in environment before running.")

# -------------------------
# Main columns
# -------------------------
col1, col2 = st.columns([2, 1])

with col1:
    question = st.text_area("Ask a question about EFDA medical guidelines:", height=120, placeholder="e.g., How shall the promotion of medicines be conducted?")
    col1a, col1b = st.columns(2)
    with col1a:
        run_qa = st.button("Search & Answer", type="primary", use_container_width=True)
    with col1b:
        run_summary = st.button("Summarize Documents", type="secondary", use_container_width=True)

    # -------------------------
    # Q&A flow
    # -------------------------
    if run_qa and question.strip():
        st.session_state.conversation_history.append({"role": "user", "content": question.strip()})
        filters = {"category": category} if category and category.lower() != "all" else None
        with st.spinner("Retrieving..."):
            faiss_dir = f"{chunk_file.replace('.pkl', '')}_faiss_store" if "_chunks" not in chunk_file else chunk_file.replace(".pkl", "_faiss_store")
            bm25_file = f"{chunk_file.replace('.pkl', '')}_bm25.pkl" if "_chunks" not in chunk_file else chunk_file.replace(".pkl", "_bm25.pkl")
            docs = hybrid_search_with_rerank(question.strip(), k=k, filters=filters, rerank_top_n=rerank_top_n)
        if not docs:
            st.error("No relevant documents found.")
        else:
            with st.spinner("Generating answer..."):
                llm = get_llm(model, temperature)
                chain = build_rag_chain(llm)
                result = chain.invoke({"question": question.strip(), "docs": docs})
                if isinstance(result, dict):
                    ans = result.get("response", "Error processing response")
                    context = result.get("context", "")
                    sources = result.get("sources", "")
                else:
                    ans = result.content if hasattr(result, "content") else "Error processing response"
                    context = ""
                    sources = ""
            st.session_state.conversation_history.append({"role": "assistant", "content": ans})
            st.markdown("### Answer")
            st.write(ans)
            if sources and context:
                st.markdown("### Sources")
                st.write(sources)
                st.markdown("### Retrieved Chunks")
                st.caption("Top chunks used as context (title · page · preview).")
                for i, line in enumerate(context.split("\n"), 1):
                    if line.strip():
                        st.write(line)

    # -------------------------
    # Summarization flow
    # -------------------------
    if run_summary:
        filters = {"category": category} if category and category.lower() != "all" else None
        with st.spinner("Retrieving..."):
            faiss_dir = f"{chunk_file.replace('.pkl', '')}_faiss_store" if "_chunks" not in chunk_file else chunk_file.replace(".pkl", "_faiss_store")
            bm25_file = f"{chunk_file.replace('.pkl', '')}_bm25.pkl" if "_chunks" not in chunk_file else chunk_file.replace(".pkl", "_bm25.pkl")
            docs = hybrid_search_with_rerank(question.strip() if question else "Summarize EFDA medical guidelines", k=k, filters=filters, rerank_top_n=rerank_top_n)
        if not docs:
            st.error("No relevant documents found.")
        else:
            with st.spinner("Generating summary..."):
                llm = get_llm(model, temperature)
                chain = build_summary_chain(llm)
                result = chain.invoke({"docs": docs})
                if isinstance(result, dict):
                    summary = result.get("response", "Error processing response")
                    sources = result.get("sources", "")
                else:
                    summary = result.content if hasattr(result, "content") else "Error processing response"
                    sources = ""
            st.markdown("### Summary")
            st.write(summary)
            if sources:
                st.markdown("### Sources")
                st.write(sources)

with col2:
    st.write("**Preview Panel**")
    st.caption("This panel can be used for future features like real-time chunk preview or additional insights.")
    if st.button("Enable Preview (Not Implemented)"):
        st.warning("Preview feature is not yet available.")
