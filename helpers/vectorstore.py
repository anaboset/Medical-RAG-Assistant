import pickle
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.schema import Document
from typing import List

def store_chunks(chunks: List[Document], faiss_dir: str = "./faiss_store", embeddings_model: str = "sentence-transformers/all-MiniLM-L6-v2") -> FAISS:
    """
    Stores chunks in a FAISS vectorstore with HuggingFace embeddings.
    """
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model)
    try:
        vectorstore = FAISS.from_documents(chunks, embeddings)
        vectorstore.save_local(faiss_dir)
    except Exception as e:
        raise Exception(f"Failed to store FAISS index: {e}")

def get_bm25_retriever(chunks: List[Document], bm25_file: str = "chunks_bm25.pkl") -> BM25Retriever:
    """
    Returns a BM25 retriever from given chunks (for hybrid search).
    """
    try:
        bm25 = BM25Retriever.from_documents(chunks)
        with open(bm25_file, "wb") as f:
            pickle.dump(bm25, f)
    except Exception as e:
        raise Exception(f"Failed to create BM25 retriever: {e}")

def get_vectorstore(faiss_dir: str = "./faiss_store", embeddings_model: str = "sentence-transformers/all-MiniLM-L6-v2") -> FAISS:
    """
    Loads FAISS vectorstore from disk.
    """
    try:
        embeddings = HuggingFaceEmbeddings(model_name=embeddings_model)
        return FAISS.load_local(faiss_dir, embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        raise Exception(f"Failed to load FAISS index: {e}")

def get_bm25(bm25_file: str = "chunks_bm25.pkl") -> BM25Retriever:
    """
    Loads BM25 retriever from disk.
    """
    try:
        with open(bm25_file, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        raise Exception(f"Failed to load BM25 retriever: {e}")
