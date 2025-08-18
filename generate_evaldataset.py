import pickle
import random
import json
from collections import defaultdict
from helpers.vectorstore import get_vectorstore, get_bm25
from langchain.retrievers import EnsembleRetriever
from langchain_groq import ChatGroq

CHUNKS_FILE = "chunks.pkl"
OUTPUT_FILE = "eval_dataset.json"
NUM_QS_PER_CATEGORY = 5  # how many Q&As per category

# ---- Load Chunks ----
def load_chunks():
    with open(CHUNKS_FILE, "rb") as f:
        return pickle.load(f)

# ---- Group Chunks by Category ----
def group_chunks_by_category(chunks):
    grouped = defaultdict(list)
    for doc in chunks:
        category = doc.metadata.get("category", "uncategorized")
        grouped[category].append(doc)
    return grouped

# ---- LLM for Q&A ----
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1)

def make_qa_from_chunk(doc):
    context = doc.page_content.strip().replace("\n", " ")
    prompt = f"""
Create a **single** question and its direct, concise answer based ONLY on this text:

{context}

Rules:
- The answer must be one sentence or short phrase **directly from the text**.
- Avoid vague or opinion-based questions.
- The question must test factual recall of the text.

Respond ONLY in valid JSON format:
{{"question": "...", "answer": "..."}}
""".strip()
    try:
        response = llm.invoke(prompt).content.strip()
        return json.loads(response)
    except Exception as e:
        print(f"⚠️ Skipped chunk due to error: {e}")
        return None

if __name__ == "__main__":
    print("📥 Loading chunks...")
    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks.")

    grouped_chunks = group_chunks_by_category(chunks)
    eval_data = []

    for category, docs in grouped_chunks.items():
        print(f"📚 Category: {category} ({len(docs)} chunks)")

        sample_count = min(NUM_QS_PER_CATEGORY, len(docs))
        sampled = random.sample(docs, sample_count)

        for doc in sampled:
            qa = make_qa_from_chunk(doc)
            if qa:
                eval_data.append(qa)

    with open(OUTPUT_FILE, "w") as f:
        json.dump(eval_data, f, indent=2)

    print(f"✅ Saved {len(eval_data)} Q&A pairs to {OUTPUT_FILE}")
