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
    Perform hybrid search by combining FAISS (dense) and BM25 (sparse) retrievers.

    Args:
        query (str): Search query string.
        top_n (int, optional): Maximum number of results to retrieve from
            each retriever. Defaults to 20.
        filters (Optional[Dict], optional): Metadata filters for FAISS retriever.
            Defaults to None.

    Returns:
        List[Document]: A merged and deduplicated list of retrieved documents.
    """
    faiss_store = get_vectorstore()
    bm25 = get_bm25()

    # Retrieve top-N documents from FAISS 
    dense_docs = faiss_store.as_retriever(search_kwargs={"k": top_n, "filter": filters}).invoke(query)
    # Retrieve top-N documents from BM25
    sparse_docs = bm25.invoke(query)[:top_n]

    # Merge and deduplicate by (content + title + page) to avoid redundancy
    merged = []
    seen = set()
    for doc in dense_docs + sparse_docs:
        key = (
            doc.page_content, 
            doc.metadata.get("doc_title"), 
            doc.metadata.get("page")
        )
        if key not in seen:
            seen.add(key)
            merged.append(doc)

    return merged

def rerank(query: str, docs: List[Document]) -> List[Document]:
    """
    Rerank retrieved documents by semantic relevance using a cross-encoder.

    Args:
        query (str): The search query string.
        docs (List[Document]): List of candidate documents.

    Returns:
        List[Document]: Documents sorted by descending relevance score.
    """
    reranker = load_reranker()
    # Create pairs of (query, document content) for reranking
    pairs = [(query, d.page_content) for d in docs]
    scores = reranker.predict(pairs)

    # Sort documents by relevance score (highest first)
    return [
        doc for _, doc in sorted(
            zip(scores, docs), 
            key=lambda x: x[0], 
            reverse=True)]

def hybrid_search_with_rerank(
        query: str, 
        k: int = 6, 
        filters: Optional[dict] = None, 
        rerank_top_n: int = 20
        ) -> List[Document]:
    """
    Hybrid search pipeline with reranking.

    Args:
        query (str): The search query string.
        k (int, optional): Number of top reranked documents to return. Defaults to 6.
        filters (Optional[Dict], optional): Metadata filters for FAISS retriever. Defaults to None.
        rerank_top_n (int, optional): Number of documents to retrieve before reranking.
            Defaults to 20.

    Returns:
        List[Document]: Top-k reranked documents most relevant to the query.
    """
    # Step 1: Perform hybrid retrieval
    docs = hybrid_search(query, top_n=rerank_top_n, filters=filters)

    # Step 2: Rerank retrieved documents by semantic relevance
    reranked = rerank(query, docs)

    # Step 3: Return top-k reranked documents
    return reranked[:k]