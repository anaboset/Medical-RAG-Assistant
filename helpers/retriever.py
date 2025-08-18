from helpers.vectorstore import get_vectorstore, get_bm25
from sentence_transformers import CrossEncoder
from langchain.schema import Document
from typing import List, Optional

def load_reranker():
    """
    Load cross-encoder for reranking.
    """
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def hybrid_search(query: str, top_n: int = 20, filters: Optional[dict] = None) -> List[Document]:
    """
    Retrieve top N docs from FAISS (dense) and BM25 (sparse), merge, and deduplicate.
    """
    faiss_store = get_vectorstore()
    bm25 = get_bm25()

    dense_docs = faiss_store.as_retriever(search_kwargs={"k": top_n, "filter": filters}).invoke(query)
    sparse_docs = bm25.invoke(query)[:top_n]

    # Merge and deduplicate by page content to avoid redundancy
    merged = []
    seen = set()
    for doc in dense_docs + sparse_docs:
        key = (doc.page_content, doc.metadata.get("doc_title"), doc.metadata.get("page"))
        if key not in seen:
            seen.add(key)
            merged.append(doc)

    return merged

def rerank(query: str, docs: List[Document]) -> List[Document]:
    """
    Rerank documents using cross-encoder based on query relevance.
    """
    reranker = load_reranker()
    pairs = [(query, d.page_content) for d in docs]
    scores = reranker.predict(pairs)
    return [doc for _, doc in sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)]

def hybrid_search_with_rerank(query: str, k: int = 6, filters: Optional[dict] = None, rerank_top_n: int = 20) -> List[Document]:
    """
    Wrapper for hybrid search with reranking: Retrieve top N docs and return top k after reranking.
    """
    docs = hybrid_search(query, top_n=rerank_top_n, filters=filters)
    reranked = rerank(query, docs)
    return reranked[:k]