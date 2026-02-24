"""Document ingestion: chunk, embed, and store in ChromaDB."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import (
    CHROMA_COLLECTION_NAME,
    CHROMA_PERSIST_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL,
    INGEST_BATCH_SIZE,
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


def ingest_documents(docs: list[Document]) -> Chroma:
    """
    Chunk, embed, and store pre-loaded Documents in ChromaDB.

    Useful for programmatic sources (e.g. HubSpot) where documents are already
    in memory rather than on disk.

    Args:
        docs: List of LangChain Document objects.

    Returns:
        Chroma vectorstore instance (persisted to disk).
    """
    embeddings = get_embeddings()
    text_splitter = get_text_splitter()

    splits = text_splitter.split_documents(docs)

    CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)

    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=str(CHROMA_PERSIST_DIR),
        collection_name=CHROMA_COLLECTION_NAME,
    )

    return vectorstore


def ingest_documents_batched(
    docs: list[Document],
    batch_size: int | None = None,
    on_progress: Callable[[int, int, str], None] | None = None,
) -> Chroma:
    """
    Chunk, embed, and store Documents in ChromaDB in batches.

    Reduces memory use and avoids timeouts for large datasets (e.g. 40K+ contacts).
    Replaces any existing collection with the same name.

    Args:
        docs: List of LangChain Document objects.
        batch_size: Docs per batch (default from INGEST_BATCH_SIZE).
        on_progress: Optional callback(processed, total, message) for UI/logging.

    Returns:
        Chroma vectorstore instance (persisted to disk).
    """
    if not docs:
        raise ValueError("No documents to ingest")

    batch_size = batch_size or INGEST_BATCH_SIZE
    embeddings = get_embeddings()
    text_splitter = get_text_splitter()

    CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)

    # Delete existing collection for replace semantics
    try:
        client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))
        client.delete_collection(CHROMA_COLLECTION_NAME)
    except Exception:
        pass

    vectorstore = Chroma(
        persist_directory=str(CHROMA_PERSIST_DIR),
        embedding_function=embeddings,
        collection_name=CHROMA_COLLECTION_NAME,
    )

    total = len(docs)
    total_batches = (total + batch_size - 1) // batch_size
    processed = 0

    for i in range(0, total, batch_size):
        batch = docs[i : i + batch_size]
        batch_num = (i // batch_size) + 1
        splits = text_splitter.split_documents(batch)
        vectorstore.add_documents(splits)
        processed += len(batch)
        msg = f"Embedding batch {batch_num}/{total_batches} ({processed:,}/{total:,} docs)"
        if on_progress:
            on_progress(processed, total, msg)

    return vectorstore


def _delete_collection_if_corrupt() -> None:
    """Delete the Chroma collection so it can be rebuilt cleanly."""
    try:
        client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))
        client.delete_collection(CHROMA_COLLECTION_NAME)
        print(f"[Chroma] Deleted corrupt collection '{CHROMA_COLLECTION_NAME}' — re-index required.")
    except Exception:
        pass


def load_vectorstore() -> Chroma | None:
    """
    Load existing ChromaDB vectorstore from disk.

    On corruption (InternalError), deletes the broken collection and returns None
    so callers know to re-index.

    Returns:
        Chroma vectorstore if it exists and is healthy, else None.
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
        # Probe the collection with a trivial count to surface corruption early
        vectorstore._collection.count()
        return vectorstore
    except chromadb.errors.InternalError as e:
        print(f"[Chroma] Corrupt index detected: {e}. Deleting and returning None.")
        _delete_collection_if_corrupt()
        return None
    except Exception:
        return None
