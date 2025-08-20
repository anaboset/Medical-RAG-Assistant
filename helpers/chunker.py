from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
from typing import List

def chunk_documents(docs: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    """
    Split documents into chunks with metadata preservation.

    Args:
        docs: A list of Document objects to be chunked.
        chunk_size: The maximum size of each chunk. Defaults to 1000.
        chunk_overlap: The number of characters to overlap between adjacent chunks. Defaults to 200.

    Returns:
        A new list of Document objects, where each object is a text chunk with its original metadata.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]  # Prioritize splitting by paragraphs, then lines, then words, then characters
    )
    chunks = []
    for doc in docs:
        split_texts = splitter.split_text(doc.page_content)

        # Create a new Document object for each text chunk,
        # preserving the original metadata from the source document
        for text in split_texts:
            chunks.append(Document(
                page_content=text,
                metadata=doc.metadata  # Preserve category, doc_title, page
            ))
    return chunks 