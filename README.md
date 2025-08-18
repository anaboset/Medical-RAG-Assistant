EFDA Guidelines Q&A with RAG 💬
This is an interactive Q&A application that allows you to query EFDA (European Federation of Dental Associations) guidelines. It uses a Retrieval-Augmented Generation (RAG) pipeline to retrieve relevant guideline snippets and provide accurate answers based on the processed documents.
[GIF of the app in action]
Features

Interactive Q&A: Ask questions about EFDA guidelines in natural language.
Advanced Retrieval: Uses hybrid search (FAISS + BM25) with a cross-encoder reranker for highly accurate context retrieval.
Fast Generation: Powered by the Groq API with Llama 3.3 70B for near-instant answers.
Open-Source Embeddings: Utilizes a local Hugging Face model (all-MiniLM-L6-v2) for text embeddings.
Simple UI: Built with Streamlit for a clean, user-friendly web interface.
Multiple Document Sets: Switch between different PDF collections via chunk file selection.
Summarization Mode: Summarize retrieved documents with a single click.

Tech Stack

Framework: LangChain
UI: Streamlit
LLM: Groq (Llama 3.3 70B Versatile)
Embedding Model: Hugging Face sentence-transformers/all-MiniLM-L6-v2
Vector Store: FAISS (dense) + BM25 (sparse)
Reranker: cross-encoder/ms-marco-MiniLM-L-6-v2
Data Loader: PyPDF2

Getting Started
Follow these instructions to set up and run the project on your local machine.
Prerequisites

Python 3.8 or higher
Git

1. Clone the Repository
Clone the project repository to your local machine.
git clone https://github.com/your-username/efda-rag-assistant.git
cd efda-rag-assistant

2. Create a Python Virtual Environment
It’s highly recommended to use a virtual environment to manage project dependencies.
# Create the virtual environment
python3 -m venv venv

# Activate it
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate

3. Install Dependencies
The project’s dependencies are listed in requirements.txt.
pip install -r requirements.txt

4. Set Up Environment Variables
The application requires an API key from Groq to use its LLM.
Create a file named .env in the root of your project directory:
cp .env.example .env

(If you don’t have a .env.example file, create a new file named .env.)
Get your API key from the GroqCloud Console.
Open the .env file and add your API key:
GROQ_API_KEY="gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

Preprocessing
Before running the app, process PDFs to create chunks and indexes:

Organize PDFs:

Place PDF files in a directory structure like /path/to/data/<category>/file.pdf.
Example: /media/matinol/Demal/data/regulations/doc1.pdf.


Run preprocessing:
python process_pdfs.py


This loads PDFs, chunks them, and creates chunks.pkl, FAISS index (./chunks_faiss_store), and BM25 index (chunks_bm25.pkl).
Default data path: /media/matinol/Demal/data (edit process_pdfs.py if needed).
For multiple document sets, run with a custom prefix:python process_pdfs.py --output_prefix chunks_set1





Usage
With the environment set up and indexes created, run the Streamlit application:
streamlit run app.py

Your web browser should automatically open with the application running.

Configure settings in the sidebar:
Document set: Select a chunk file (e.g., chunks.pkl, chunks_set1.pkl).
Groq model: Default is llama-3.3-70b-versatile.
Temperature: Adjust LLM creativity (0.0–1.0).
Top-k results: Number of documents to return (3–12).
Rerank Top N: Number of documents to rerank (10–50).
Category filter: Select a category or "all".


Enter a question (e.g., "What is the process for variation applications to registered medicines?") or click "Summarize Documents" for a summary.
Click "Search & Answer" or "Summarize Documents" to view the response and retrieved chunks.

Example Queries and Answers

Query: "What is the process for variation applications to registered medicines?"

Answer: The process for variation applications to registered medicines involves submitting a formal request to the regulatory authority, including updated documentation and evidence of compliance with EFDA guidelines. Variations are classified as minor (Type IA/IB) or major (Type II), with specific requirements for each. [Details depend on retrieved chunks.]
Sources: 
Guideline on Variations p.5
Regulatory Procedures p.12




Query: "Summarize EFDA guidelines on dental equipment standards."

Answer: The EFDA guidelines on dental equipment standards emphasize compliance with safety, performance, and maintenance requirements. Equipment must meet CE marking standards, undergo regular inspections, and adhere to sterilization protocols to ensure patient safety. [Summary based on retrieved chunks.]
Sources: 
Dental Equipment Standards p.3
Safety Protocols p.7





Limitations and Known Issues

PDF Parsing: Some PDFs with complex formatting (e.g., tables, images) may not parse correctly with PyPDF2, leading to incomplete text extraction.
Reranking Speed: Cross-encoder reranking (ms-marco-MiniLM-L-6-v2) can be slow for large rerank_top_n values (>30), especially on CPU-only systems.
Category Dependency: Category filtering requires PDFs to be organized in a folder structure; unorganized PDFs default to "uncategorized."
Local Execution: Large PDF collections may require significant memory for preprocessing and indexing.

Project Structure
The project is organized in a modular way to keep the code clean and maintainable.
.
├── helpers/
│   ├── __init__.py       # Makes 'helpers' a Python package
│   ├── chain.py          # Builds the RAG and summary chains with the LLM
│   ├── chunker.py        # Splits documents into smaller chunks
│   ├── pdfloader.py      # Loads and processes PDF guidelines
│   ├── retriever.py      # Performs hybrid search and reranking
│   └── vectorstore.py    # Creates/loads FAISS and BM25 indexes
├── .env                  # Stores API keys (secret, not committed to git)
├── .gitignore            # Specifies files for git to ignore
├── app.py                # The main Streamlit application file
├── process_pdfs.py       # Preprocesses PDFs and creates indexes
├── requirements.txt      # Project dependencies
└── README.md             # This file

How It Works
The application follows a standard RAG pipeline:

Ingestion: The pdfloader loads PDF guidelines using PyPDF2 and adds metadata (category, title, page).
Chunking: The chunker splits documents into smaller, overlapping chunks.
Indexing: The vectorstore uses all-MiniLM-L6-v2 to create embeddings for FAISS and indexes text for BM25.
Retrieval & Reranking: The retriever performs hybrid search (FAISS + BM25), deduplicates results, and reranks using the cross-encoder for higher accuracy.
Generation: The chain passes top-ranked chunks and the question to the Groq LLM within a structured prompt to generate a grounded answer or summary.
