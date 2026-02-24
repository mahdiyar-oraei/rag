"""RAG chain: retriever + prompt + ChatOpenAI."""

from langchain_openai import ChatOpenAI
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import VectorStoreRetriever

from .config import LLM_MODEL, OPENAI_API_KEY, TOP_K


def create_contact_scoped_retriever(vectorstore, contact_id: str, k: int | None = None) -> VectorStoreRetriever:
    """
    Create a retriever that only returns documents for a specific contact:
    - The contact's own record (hs_object_id == contact_id)
    - Deals associated with that contact (associated_contact_id == contact_id)
    """
    cid = str(contact_id)
    return vectorstore.as_retriever(
        search_kwargs={
            "k": k or TOP_K,
            "filter": {
                "$or": [
                    {"hs_object_id": cid},
                    {"associated_contact_id": cid},
                ]
            },
        }
    )


def answer_for_contact(vectorstore, contact_id: str, query: str) -> str:
    """
    Run contact-scoped RAG: answer the query using only that contact's data and their deals.
    """
    retriever = create_contact_scoped_retriever(vectorstore, contact_id)
    rag_chain = create_rag_chain(retriever)
    result = rag_chain.invoke({"input": query})
    return (result.get("answer") or "").strip() or "I couldn't find relevant information for your account."


def get_llm() -> ChatOpenAI:
    """Create ChatOpenAI instance."""
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is not set. Add it to your .env file.")
    return ChatOpenAI(model=LLM_MODEL, temperature=0)


def create_rag_chain(retriever: VectorStoreRetriever):
    """
    Build the RAG chain: retriever + prompt + LLM.

    Args:
        retriever: Vector store retriever (from Chroma).

    Returns:
        Invokable RAG chain.
    """
    llm = get_llm()

    system_prompt = (
        "You are a CRM sales intelligence assistant. "
        "The context contains CRM records from HubSpot: Contacts, Companies, Deals, and Owners. "
        "Treat 'account' as a synonym for Company, 'deal size' as Amount, and 'rep' as Owner. "
        "Synthesize information across record types to give a complete answer. "
        "If partial information exists, share what you know and note what is missing. "
        "Only say you don't know if the context contains no relevant information at all. "
        "Do not invent data not present in the context.\n\nContext: {context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    return rag_chain


def correct_query(query: str) -> str:
    """Fix typos and clarify CRM intent using the LLM before retrieval."""
    if not query.strip():
        return query
    llm = get_llm()
    result = llm.invoke(
        "Fix any typos and rephrase the following CRM question for clarity. "
        "Return only the corrected question, no explanation:\n\n" + query
    )
    return (result.content or "").strip() or query


def get_retriever(vectorstore, k: int | None = None) -> VectorStoreRetriever:
    """Create retriever from vectorstore with top-k results."""
    return vectorstore.as_retriever(search_kwargs={"k": k or TOP_K})
