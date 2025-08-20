# **Medical RAG Assistant for EFDA Guidelines 💬**

This is an interactive Q&A application that allows you to query **EFDA (Ethiopian Food and Drug Authority)** medical guidelines for medicine registration, import, and export regulations. It uses a **Retrieval-Augmented Generation (RAG)** pipeline to retrieve relevant guideline snippets and provide accurate, context-aware answers based on preloaded or optionally uploaded PDF documents.

## **Features**

- **Interactive Q&A**: Ask questions about EFDA medical guidelines in natural language.
- **Context-Aware Answers**: Understands the meaning and intent behind questions, not just keywords.
- **Conversation Memory**: Maintains context across multiple questions within a session.
- **Advanced Retrieval**: Uses hybrid search (**FAISS + BM25**) with a cross-encoder reranker for highly accurate context retrieval.
- **Fast Generation**: Powered by the **Groq API** with **Llama 3.3 70B** for near-instant answers.
- **Open-Source Embeddings**: Utilizes a local **Hugging Face model (all-MiniLM-L6-v2)** for text embeddings.
- **Simple UI**: Built with **Streamlit** for a clean, user-friendly web interface.
- **Multiple Document Sets**: Switch between preloaded EFDA data or user-uploaded sets via chunk file selection.
- **Summarization Mode**: Summarize retrieved medical documents with a single click.
- **Incremental Updates**: Add new medical PDFs without reprocessing existing preloaded data (optional uploads).
- **User Uploads**: Optionally upload your own medical PDFs to generate custom indexes.

## **Tech Stack**

- **Framework**: LangChain
- **UI**: Streamlit
- **LLM**: Groq (Llama 3.3 70B Versatile)
- **Embedding Model**: Hugging Face sentence-transformers/all-MiniLM-L6-v2
- **Vector Store**: FAISS (dense) + BM25 (sparse)
- **Reranker**: cross-encoder/ms-marco-MiniLM-L-6-v2
- **Data Loader**: PyPDF2

## **Getting Started**

Follow these instructions to set up and run the project locally or deploy to **Streamlit Cloud**.

### **Prerequisites**

- **Python 3.8 or higher**
- **Git**

### **1. Clone the Repository**

Clone the project repository to your local machine.

```bash
git clone https://github.com/anaboset/Medical-RAG-Assistant
cd Medical-RAG-Assistant
```

### 2. Create a Python Virtual Environment
It’s highly recommended to use a virtual environment to manage project dependencies.
bash# Create the virtual environment
python3 -m venv venv

### Activate it
 On macOS/Linux:
source venv/bin/activate

 On Windows:
venv\Scripts\activate

### 3. Install Dependencies
The project’s dependencies are listed in `requirements.txt`.
```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables
The application requires an API key from Groq to use its LLM.
- Create a file named `.env` in the root of your project directory:
  ```bash
  cp .env.example .env
  ```
  (If you don’t have a `.env.example` file, create a new file named `.env`.)
- Get your API key from the [GroqCloud Console](https://console.groq.com/).
- Open the `.env` file and add your API key:
  ```
  GROQ_API_KEY="gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
  ```

## Preprocessing
The app includes preprocessed data (`chunks.pkl`, `faiss_store`, `bm25.pkl`) from EFDA medical PDFs, allowing it to run out of the box without requiring users to provide files. Optionally, users can upload their own medical PDFs for processing.

### Organize Medical PDFs (Optional):
- Place medical PDF files in a directory structure like `/path/to/data/<category>/file.pdf` if preprocessing locally.
- Alternatively, upload PDFs via the app's sidebar interface to generate new indexes (`uploaded_chunks.pkl`).
- To preprocess manually, run `python process_pdfs.py --input_dir data --output_prefix chunks` with PDFs in the `data` directory.

### Run Preprocessing:
- **For a full reprocess** (includes all medical PDFs, recommended initially or after updating existing PDFs):
  ```bash
  python process_pdfs.py
  ```
  Creates `chunks.pkl`, `chunks_faiss_store`, `chunks_bm25.pkl`.
  Default data path: Uses pre-included EFDA PDFs (edit `process_pdfs.py` for custom paths or use `--data_root`).

- **For a new document set** (e.g., separate medical PDF collection):
  ```bash
  python process_pdfs.py --output_prefix chunks_set1
  ```
  Creates `chunks_set1.pkl`, `chunks_set1_faiss_store`, `chunks_set1_bm25.pkl`.

### Verify Output Files:
- Ensure `chunks.pkl`, `chunks_faiss_store`, and `chunks_bm25.pkl` (or equivalents with `--output_prefix`) exist in the project root.
- If deploying to Streamlit Cloud, include these files in the repo or configure external storage (see Deployment section).

## Deploying to Streamlit Cloud
- Push your repository to GitHub with preprocessed indexes (`chunks.pkl`, `chunks_faiss_store/`, `chunks_bm25.pkl`, etc.).
- Connect the repo to [Streamlit Cloud](https://streamlit.io/cloud).
- Add `GROQ_API_KEY` as a secret in the Streamlit Cloud dashboard.
- Deploy the app. The URL will be provided (e.g., `https://your-app-name.streamlit.app`).

## Using the App
### Local Usage
With the environment set up and indexes created, run the Streamlit application:
```bash
streamlit run app.py
```

### Cloud Usage
Visit the deployed URL (e.g., `https://your-app-name.streamlit.app`).

#### Preloaded Medical PDFs
- The app uses preprocessed EFDA data by default. No upload is required initially.

#### User Interaction
- **Configure settings in the sidebar:**
  - **Document set**: Select a preloaded set (e.g., `chunks.pkl`) or an uploaded set (e.g., `uploaded_chunks.pkl`) after processing.
  - **Groq model**: Default is `llama-3.3-70b-versatile`.
  - **Temperature**: Adjust LLM creativity (0.0–1.0).
  - **Top-k results**: Number of documents to return (3–12).
  - **Rerank Top N**: Number of documents to rerank (10–50).
  - **Category filter**: Select a medical category or "all".
- **Enter a medical question** (e.g., "When can the importation of medical products by firms other than health institutions be considered??") or click "Summarize Documents" for a summary.
- Click "Search & Answer" or "Summarize Documents" to view the response and retrieved chunks.
- The app remembers your questions and answers within a session, allowing follow-ups to build on prior context.
- **Uploading Medical PDFs**: Optionally upload PDFs via the sidebar to generate custom indexes (`uploaded_chunks.pkl`). Clear the cache after processing to use the new data.

### Example Queries and Answers
- **Query**: "What is the process for registering a new medicine in Ethiopia?"
  - **Answer**: The process for registering a new medicine in Ethiopia involves identifying a local agent or representative, submitting a registration application with detailed information about the medicine, including clinical study reports, and obtaining approval from the EFDA.
    [Details depend on retrieved chunks.]
  - **Sources**: 
    - `guidelines-for-registration-of-medicine p.23`
    - `guideline variation application registered medicines p.4`

- **Follow-up**: "What about import regulations?"
  - **Answer**: Import regulations require a permit from EFDA, compliance with GMP, and labeling in English or Amharic. Priority is given to essential medicines, with inspections at ports of entry.
  - **Sources**: 
    - `guideline special import permit p.5`
    - `guideline special import permit p.17`
  - **Note**: The answer builds on the prior registration question’s context.

- **Query**: "Summarize EFDA guidelines on improving agricultural yield"
  - **Answer**: I could not find relevant information in the EFDA medical guidelines provided.

## Limitations and Known Issues
- **Medical Advice**: The app provides information from EFDA medical guidelines only and is not a substitute for professional medical advice.
- **PDF Parsing**: Some medical PDFs with complex formatting (e.g., tables, images) may not parse correctly with PyPDF2, leading to incomplete text extraction.
- **Reranking Speed**: Cross-encoder reranking (ms-marco-MiniLM-L-6-v2) can be slow for large `rerank_top_n` values (>30), especially on CPU-only systems.
- **Category Dependency**: Category filtering requires medical PDFs to be organized in a folder structure; unorganized PDFs default to "uncategorized."
- **Local Execution**: Large medical PDF collections may require significant memory for preprocessing and indexing.
- **Incremental Updates**: Incremental updates are not supported; run full preprocessing (`python process_pdfs.py`) to ensure all changes are captured.
- **Upload Limitations**: Temporary storage limits on Streamlit Cloud may affect large uploads.
- **Memory Scope**: Conversation memory is session-based and resets when the cache is cleared or the app restarts.

## Project Structure
The project is organized in a modular way to keep the code clean and maintainable.
```
.
├── helpers/
│   ├── __init__.py       # Makes 'helpers' a Python package
│   ├── chain.py          # Builds the RAG and summary chains with the LLM
│   ├── chunker.py        # Splits documents into smaller chunks
│   ├── pdfloader.py      # Loads and processes medical PDF guidelines
│   ├── retriever.py      # Performs hybrid search and reranking
│   └── vectorstore.py    # Creates/loads FAISS and BM25 indexes
├── .env                  # Stores API keys (secret, not committed to git)
├── .gitignore            # Specifies files for git to ignore
├── app.py                # The main Streamlit application file
├── process_pdfs.py       # Preprocesses medical PDFs and creates indexes
├── requirements.txt      # Project dependencies
├── README.md             # This file
```

## How It Works
The application follows a standard RAG pipeline:
- **Ingestion**: The `pdfloader` loads medical PDF guidelines using PyPDF2 and adds metadata (category, title, page).
- **Chunking**: The `chunker` splits documents into smaller, overlapping chunks.
- **Indexing**: The `vectorstore` uses all-MiniLM-L6-v2 to create embeddings for FAISS and indexes text for BM25.
- **Retrieval & Reranking**: The `retriever` performs hybrid search (FAISS + BM25), deduplicates results, and reranks using the cross-encoder for higher accuracy.
- **Generation**: The `chain` passes top-ranked chunks and the question (with conversation history) to the Groq LLM within a structured prompt to generate a context-aware medical answer or summary.

## Future Improvements
- **Knowledge Graph Integration**: Develop a domain-specific knowledge graph using Neo4j or GraknAI to define entities (e.g., medicines, regulations) and relations (e.g., approval processes), enhancing retrieval with structured medical semantics.
- **Dense Passage Retrieval (DPR) Optimization**: Fine-tune DPR models from Hugging Face Transformers on EFDA medical data to improve semantic understanding.
- **Query Rewriting and Expansion**: Implement T5 or GPT-based contextual query rewriting to resolve ambiguities in medical queries.
- **Iterative Retrieval with Feedback**: Introduce iterative retrieval rounds with user feedback to enhance query performance.
- **Contextual Compression**: Apply PEGASUS or BART for post-retrieval compression, preserving essential medical context.
- **Retrieval-Augmented Fine-Tuning**: Fine-tune the LLM with retrieval-augmented examples for broader applicability.
- **Multilingual Support**: Extend to Amharic, Swahili, and other East African languages using translation models.
- **Regional Expansion**: Incorporate guidelines from East African Community (EAC) countries.
- **Real-Time Regulatory Updates**: Integrate an API to fetch and process new medical regulations periodically.

```
