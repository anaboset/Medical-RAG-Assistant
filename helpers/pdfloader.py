import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document
from typing import List

def load_pdfs(folder_path: str, category: str) -> List[Document]:
    """
    Loads PDFs from a folder and adds metadata (category, title, page).

    Args:
        folder_path: The path to the folder containing the PDF files.
        category: The category to assign to all loaded documents.
    Returns:
        A list of Document objects, each representing a page from the PDFs with added metadata.
    """
    docs = []

    # Iterate over all files in the specified folder
    for file in os.listdir(folder_path):
        if not file.lower().endswith(".pdf"):   # Skep non-PDF files (case-insensitive)
            continue
        pdf_path = os.path.join(folder_path, file)
        try:
            pdf_docs = PyPDFLoader(pdf_path).load()   # Load the pages of the PDF as a list of Document objects using PyPDFLoader
            
            # Iterate through each page (Document) loaded from the current PDF
            for doc in pdf_docs:
                # Add or update metadata for each document (page)
                doc.metadata.update({
                    "category": category,
                    "doc_title": file.replace("_", " ").replace(".pdf", ""),
                    "page": doc.metadata.get("page")  # Preserve the original page number
                })

            # Extend the main docs list with processed pages    
            docs.extend(pdf_docs)
        except Exception as e:
            # Handle any errors that occur during PDF loading
            print(f"Error loading {pdf_path}: {e}")
    return docs
