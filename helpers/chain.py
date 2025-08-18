from langchain_groq import ChatGroq
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from typing import List

def format_context(docs: List[Document], max_snippet_len: int = 200) -> str:
    """
    Format document chunks for prompt context.
    """
    lines = []
    for d in docs:
        title = d.metadata.get("doc_title", "Unknown")
        page = d.metadata.get("page")
        page_str = f" p.{page}" if page is not None else ""
        snippet = d.page_content.strip().replace("\n", " ")[:max_snippet_len]
        lines.append(f"- [{title}{page_str}] {snippet}...")
    return "\n".join(lines) or "- (no context)"

def format_sources(docs: List[Document]) -> str:
    """
    Format document sources for prompt.
    """
    seen = set()
    sources = []
    for d in docs:
        title = d.metadata.get("doc_title", "Unknown")
        page = d.metadata.get("page")
        entry = f"{title} p.{page}" if page is not None else title
        if entry not in seen:
            seen.add(entry)
            sources.append(f"- {entry}")
    return "\n".join(sources) or "- (no sources)"

def build_rag_chain(llm: ChatGroq) -> RunnableSequence:
    """
    Build RAG chain for Q&A with PromptTemplate and LLM.
    """
    prompt = PromptTemplate.from_template("""
You are an EFDA assistant.
Use the provided context to answer the question accurately.
If the context is clearly unrelated, say "I could not find relevant information in the EFDA guidelines provided."

Context:
{context}

Q: {question}
A:

After answering, include this section:

Sources:
{sources}
""")
    return RunnableSequence(
        {
            "context": lambda x: format_context(x["docs"]),
            "sources": lambda x: format_sources(x["docs"]),
            "question": lambda x: x["question"]
        },
        prompt,
        llm
    )

def build_summary_chain(llm: ChatGroq) -> RunnableSequence:
    """
    Build chain to summarize retrieved documents.
    """
    prompt = PromptTemplate.from_template("""
You are an EFDA assistant.
Summarize the following context from EFDA guidelines in a concise paragraph (100-150 words).
Focus on key points and avoid including unnecessary details.

Context:
{context}

Summary:

Sources:
{sources}
""")
    return RunnableSequence(
        {
            "context": lambda x: format_context(x["docs"]),
            "sources": lambda x: format_sources(x["docs"])
        },
        prompt,
        llm
    )
