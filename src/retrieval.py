"""RAG chain: retriever + prompt + ChatOpenAI."""

from langchain_openai import ChatOpenAI
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import VectorStoreRetriever

from .config import LLM_MODEL, OPENAI_API_KEY, TOP_K


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
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer based on the context, say that you don't know. "
        "Do not make up information.\n\nContext: {context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    return rag_chain


def get_retriever(vectorstore, k: int | None = None) -> VectorStoreRetriever:
    """Create retriever from vectorstore with top-k results."""
    return vectorstore.as_retriever(search_kwargs={"k": k or TOP_K})
