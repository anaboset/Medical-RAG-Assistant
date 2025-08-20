from langchain_groq import ChatGroq
from langchain.schema import Document, AIMessage
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence, RunnableLambda, RunnablePassthrough
from typing import List, Optional, Dict

def format_context(docs: List[Document], max_snippet_len: int = 200) -> str:
    """
    Format document chunks into a readable context string for inclusion in prompts.

    Args:
        docs (List[Document]): A list of LangChain Document objects, each containing
            page content and metadata.
        max_snippet_len (int, optional): Maximum number of characters to include
            from each document's text snippet. Defaults to 200.

    Returns:
        str: A string containing formatted document snippets, each prefixed with its
        title and page number (if available). Returns "- (no context)" if no documents
        are provided.
    """
    lines = []
    
    for d in docs:  # Iterate over Document objects directly
        if isinstance(d, Document):
            title = d.metadata.get("doc_title", "Unknown")
            page = d.metadata.get("page")
            page_str = f" p.{page}" if page is not None else ""
            
            # Extract snippet from page content with newlines replaced
            snippet = d.page_content.strip().replace("\n", " ")[:max_snippet_len]
            lines.append(f"- [{title}{page_str}] {snippet}...")
    return "\n".join(lines) or "- (no context)"


def format_sources(docs: List[Document]) -> str:
    """
    Extract and format unique sources from documents for attribution.

    Args:
        docs (List[Document]): A list of LangChain Document objects.

    Returns:
        str: A string containing formatted source titles and pages.
        Returns "- (no sources)" if none are available.
    """
    seen = set()
    sources = []

    for d in docs:  
        if isinstance(d, Document):
            title = d.metadata.get("doc_title", "Unknown")
            page = d.metadata.get("page")
            entry = f"{title} p.{page}" if page is not None else title
            if entry not in seen:
                seen.add(entry)
                sources.append(f"- {entry}")
    return "\n".join(sources) or "- (no sources)"


def build_rag_chain(llm: ChatGroq) -> RunnableSequence:
    """
    Build RAG chain for medical Q&A with PromptTemplate and LLM.

     Args:
        llm (ChatGroq): A Groq-powered LLM instance used for inference.

    Returns:
        RunnableSequence: A LangChain Runnable sequence that takes input with:
            - "docs": List[Document]
            - "question": str
        And outputs a dictionary with:
            - "response": str (answer from LLM)
            - "context": str (formatted snippets)
            - "sources": str (formatted sources)

    """
    prompt = PromptTemplate.from_template("""
You are a medical assistant specialized in EFDA (Ethiopian Food and Drug Authority) guidelines for medicine registration, import, and export regulations.
Use the provided context from EFDA medical guidelines to answer the question accurately and contextually, focusing on medical aspects.
Understand the meaning and intent behind the question, not just keywords.
If the context is clearly unrelated or insufficient, respond with only: "I could not find relevant information in the EFDA medical guidelines provided." and do not include additional sources or details.
Otherwise, provide a detailed answer.

Context:
{context}

Q: {question}
A:
""")


    def format_output(input_data: Dict) -> Dict:
        """
        Format the chain's final output by including response, context, and sources.

        Args:
            input_data (Dict[str, object]): A dictionary containing:
                - "response": AIMessage from the LLM.
                - "docs": List[Document] used as context.

        Returns:
            Dict[str, str]: Structured output containing:
                - "response": str
                - "context": str
                - "sources": str
        """
        response: AIMessage = input_data.get("response")  # Response from LLM
        content = response.content if response else "Error processing response"
        
        # If the model explicitly indicates no relevant information was found
        if "I could not find relevant information" in content:
            return {"response": content, "context": "", "sources": ""}
        else:
            return {
                "response": content,
                "context": format_context(input_data["docs"]),
                "sources": format_sources(input_data["docs"])
            }

    # Assemble the chain using LangChain Runnables
    return (
        {
            "context": lambda x: format_context(x["docs"]),
            "question": lambda x: x["question"],
            "docs": lambda x: x["docs"]
        }
        # Invoke the LLM with formatted prompt
        | RunnablePassthrough.assign(response=lambda x: llm.invoke(prompt.format(context=x["context"], question=x["question"])))
        | RunnableLambda(format_output)
    )

def build_summary_chain(llm: ChatGroq) -> RunnableSequence:
    """
    Construct a summarization chain for EFDA medical documents.

    Args:
        llm (ChatGroq): A Groq-powered LLM instance used for inference.

    Returns:
        RunnableSequence: A LangChain Runnable sequence that takes input with:
            - "docs": List[Document]
        And outputs:
            - "context": str (formatted snippets)
            - "sources": str (formatted sources)
            - "summary": str (concise summary of docs)
    """
    prompt = PromptTemplate.from_template("""
You are a medical assistant specialized in EFDA (Ethiopian Food and Drug Authority) guidelines for medicine registration, import, and export regulations.
Summarize the following context from EFDA medical guidelines in a concise paragraph (100-150 words).
Focus on key medical points and avoid including unnecessary details.
If the context is insufficient or unrelated, respond with only: "I could not find relevant information in the EFDA medical guidelines provided." and do not include additional sources or details.

Context:
{context}

Summary:
""")
    
    # Chain: extract context/sources -> build prompt -> run LLM
    return RunnableSequence(
        {
            "context": lambda x: format_context(x["docs"]),
            "sources": lambda x: format_sources(x["docs"])
        },
        prompt,
        llm
    )
