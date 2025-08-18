import os
import pickle
from helpers.pdfloader import load_pdfs
from helpers.chunker import chunk_documents
from helpers.vectorstore import store_chunks, get_bm25_retriever

def process_pdfs(data_root: str = "/media/matinol/Demal/data", output_prefix: str = "chunks"):
    """
    Load PDFs, chunk, and store FAISS/BM25 indexes with named outputs.
    """
    all_docs = []
    for cat in os.listdir(data_root):
        folder = os.path.join(data_root, cat)
        if os.path.isdir(folder):
            try:
                pdf_docs = load_pdfs(folder, cat)
                all_docs.extend(chunk_documents(pdf_docs))
            except Exception as e:
                print(f"Error processing category {cat}: {e}")
    try:
        output_file = f"{output_prefix}.pkl"
        faiss_dir = f"{output_prefix}_faiss_store"
        bm25_file = f"{output_prefix}_bm25.pkl"
        with open(output_file, "wb") as f:
            pickle.dump(all_docs, f)
        store_chunks(all_docs, faiss_dir=faiss_dir)
        get_bm25_retriever(all_docs, bm25_file=bm25_file)
    except Exception as e:
        print(f"Error saving chunks/indexes: {e}")

if __name__ == "__main__":
    process_pdfs()
