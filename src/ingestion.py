"""Document ingestion: chunk, embed, and store in ChromaDB."""

from pathlib import Path

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import (
    CHROMA_COLLECTION_NAME,
    CHROMA_PERSIST_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL,
    OPENAI_API_KEY,
)
from .loaders import load_documents


def get_embeddings() -> OpenAIEmbeddings:
    """Create OpenAI embeddings instance."""
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is not set. Add it to your .env file.")
    return OpenAIEmbeddings(model=EMBEDDING_MODEL)


def get_text_splitter() -> RecursiveCharacterTextSplitter:
    """Create text splitter with configured chunk size and overlap."""
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )


def index_documents(paths: list[str | Path]) -> Chroma:
    """
    Load, chunk, embed, and store documents in ChromaDB.

    Args:
        paths: List of file paths (PDF, MD, TXT).

    Returns:
        Chroma vectorstore instance (persisted to disk).
    """
    embeddings = get_embeddings()
    text_splitter = get_text_splitter()

    docs = load_documents(paths)
    splits = text_splitter.split_documents(docs)

    CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)

    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=str(CHROMA_PERSIST_DIR),
        collection_name=CHROMA_COLLECTION_NAME,
    )

    return vectorstore


def load_vectorstore() -> Chroma | None:
    """
    Load existing ChromaDB vectorstore from disk.

    Returns:
        Chroma vectorstore if it exists and has data, else None.
    """
    if not CHROMA_PERSIST_DIR.exists():
        return None

    embeddings = get_embeddings()

    try:
        vectorstore = Chroma(
            persist_directory=str(CHROMA_PERSIST_DIR),
            embedding_function=embeddings,
            collection_name=CHROMA_COLLECTION_NAME,
        )
        return vectorstore
    except Exception:
        return None
