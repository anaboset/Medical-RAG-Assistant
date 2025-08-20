import os
import pickle
from helpers.pdfloader import load_pdfs
from helpers.chunker import chunk_documents
from helpers.vectorstore import store_chunks, get_bm25_retriever

def process_pdfs(
        data_root: str = "./data", 
        output_prefix: str = "chunks"):
    """
     Load PDFs from categorized folders, split them into chunks, 
    and store FAISS and BM25 indexes along with pickled chunks.

    Args:
        data_root (str, optional): Root folder containing subfolders for each category.
            Defaults to "./data". If empty or no valid PDFs found, uses pre-included EFDA PDFs.
        output_prefix (str, optional): Prefix used for output files and FAISS/BM25 directories.
            Defaults to "chunks".

    Returns:
        None
    """
    all_docs = []

    # Check for user-provided PDFs in data_root or fall back to pre-included EFDA PDFs
    pdf_files = []
    if os.path.exists(data_root) and os.listdir(data_root):
        # Iterate through all categories (subfolders) in the data root
        for cat in os.listdir(data_root):
            folder = os.path.join(data_root, cat)
            if os.path.isdir(folder):
                try:
                    # Load PDFs from this category and chunk them
                    pdf_docs = load_pdfs(folder, cat)
                    all_docs.extend(chunk_documents(pdf_docs))
                except Exception as e:
                    # Log errors per category without stopping the process
                    print(f"Error processing category {cat}: {e}")
    else:
        # Fallback to pre-included EFDA PDFs if no valid data_root content
        default_pdfs = {
            "guidelines": ["data/efda_guideline1.pdf", "data/efda_guideline2.pdf"]  # Replace with your PDF paths
        }
        for cat, pdf_list in default_pdfs.items():
            for pdf_path in pdf_list:
                if os.path.exists(pdf_path):
                    try:
                        pdf_docs = load_pdfs(pdf_path, cat)  # Adjust load_pdfs to handle single file if needed
                        all_docs.extend(chunk_documents(pdf_docs))
                    except Exception as e:
                        print(f"Error processing default PDF {pdf_path}: {e}")

    if not all_docs:
        print("No PDFs processed. Using pre-existing data if available.")
        return

    try:
        # Define output file paths
        output_file = f"{output_prefix}.pkl"
        faiss_dir = f"{output_prefix}_faiss_store"
        bm25_file = f"{output_prefix}_bm25.pkl"

        # Save all_docs as a pickle file
        with open(output_file, "wb") as f:
            pickle.dump(all_docs, f)
        
        # Store FAISS index
        store_chunks(all_docs, faiss_dir=faiss_dir)

        # Store BM25 retriever
        get_bm25_retriever(all_docs, bm25_file=bm25_file)

    except Exception as e:
        # Handle any saving/indexing errors
        print(f"Error saving chunks/indexes: {e}")

if __name__ == "__main__":
    process_pdfs()
