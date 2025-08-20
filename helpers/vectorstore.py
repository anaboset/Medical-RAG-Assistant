import pickle
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.schema import Document
from typing import List


def store_chunks(chunks: List[Document], faiss_dir: str = "./faiss_store", embeddings_model: str = "sentence-transformers/all-MiniLM-L6-v2") -> FAISS:
    """
     Build and store a FAISS vectorstore using HuggingFace embeddings.

    Args:
        chunks (List[Document]): List of document chunks to index.
        faiss_dir (str, optional): Directory where FAISS index will be saved.
            Defaults to "./faiss_store".
        embeddings_model (str, optional): Name of HuggingFace embedding model to use.
            Defaults to "sentence-transformers/all-MiniLM-L6-v2".

    Returns:
        FAISS: A FAISS vectorstore object containing indexed chunks.

    Raises:
        Exception: If storing FAISS index fails.
    """
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model)
    
    
    try:
        vectorstore = FAISS.from_documents(chunks, embeddings) # Build FAISS vectorstore from documents
        vectorstore.save_local(faiss_dir) # Save FAISS index to disk for reuse
    except Exception as e:
        raise Exception(f"Failed to store FAISS index: {e}")



def get_bm25_retriever(chunks: List[Document], bm25_file: str = "chunks_bm25.pkl") -> BM25Retriever:
    """
    Build and save a BM25 retriever from document chunks.

    Args:
        chunks (List[Document]): List of document chunks to index.
        bm25_file (str, optional): File path where the retriever will be pickled.
            Defaults to "chunks_bm25.pkl".

    Returns:
        BM25Retriever: A BM25 retriever object for keyword-based retrieval.

    Raises:
        Exception: If creating or saving BM25 retriever fails.
    """
    try:
        bm25 = BM25Retriever.from_documents(chunks)
        with open(bm25_file, "wb") as f: # Save BM25 retriever to disk for reuse
            pickle.dump(bm25, f)
    except Exception as e:
        raise Exception(f"Failed to create BM25 retriever: {e}")



def get_vectorstore(faiss_dir: str = "./faiss_store", embeddings_model: str = "sentence-transformers/all-MiniLM-L6-v2") -> FAISS:
    """
    Load an existing FAISS vectorstore from disk.

    Args:
        faiss_dir (str, optional): Directory where FAISS index is stored.
            Defaults to "./faiss_store".
        embeddings_model (str, optional): Name of HuggingFace embedding model used
            when the index was created. Defaults to "sentence-transformers/all-MiniLM-L6-v2".

    Returns:
        FAISS: Loaded FAISS vectorstore object.

    Raises:
        Exception: If loading FAISS index fails.
    """
    try:
        embeddings = HuggingFaceEmbeddings(model_name=embeddings_model)
        return FAISS.load_local(faiss_dir, embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        raise Exception(f"Failed to load FAISS index: {e}")



def get_bm25(bm25_file: str = "chunks_bm25.pkl") -> BM25Retriever:
    """
     Load a BM25 retriever from disk.

    Args:
        bm25_file (str, optional): File path where the retriever was saved.
            Defaults to "chunks_bm25.pkl".

    Returns:
        BM25Retriever: A loaded BM25 retriever object.

    Raises:
        Exception: If loading BM25 retriever fails.
    """
    try:
        with open(bm25_file, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        raise Exception(f"Failed to load BM25 retriever: {e}")
