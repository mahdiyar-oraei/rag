"""Centralized configuration for the RAG application."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# OpenAI
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o-mini")

# ChromaDB
CHROMA_PERSIST_DIR: Path = Path(
    os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
).resolve()
CHROMA_COLLECTION_NAME: str = os.getenv("CHROMA_COLLECTION_NAME", "notebook_docs")

# Chunking
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))

# Retrieval
TOP_K: int = int(os.getenv("TOP_K", "15"))

# Allowed file types for upload
ALLOWED_EXTENSIONS: set[str] = {".pdf", ".md", ".txt"}

# HubSpot
HUBSPOT_ACCESS_TOKEN: str = os.getenv("HUBSPOT_ACCESS_TOKEN", "")
HUBSPOT_BASE_URL: str | None = os.getenv("HUBSPOT_BASE_URL") or None  # e.g. https://api-eu1.hubapi.com for EU
HUBSPOT_OBJECTS: list[str] = ["contacts", "companies", "deals", "owners"]
