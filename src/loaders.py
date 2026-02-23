"""Multi-format document loaders for PDF, Markdown, and plain text."""

from pathlib import Path

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader

from .config import ALLOWED_EXTENSIONS


def load_document(path: str | Path) -> list[Document]:
    """
    Load a document from the given path based on file extension.

    Supports: .pdf, .md, .txt

    Args:
        path: File path to the document.

    Returns:
        List of LangChain Document objects.

    Raises:
        ValueError: If file extension is not supported.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    suffix = path.suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type: {suffix}. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    if suffix == ".pdf":
        loader = PyPDFLoader(str(path))
    else:
        # TextLoader works for both .md and .txt
        loader = TextLoader(str(path), encoding="utf-8", autodetect_encoding=True)

    return loader.load()


def load_documents(paths: list[str | Path]) -> list[Document]:
    """
    Load multiple documents and combine their pages/chunks.

    Args:
        paths: List of file paths.

    Returns:
        Combined list of Document objects from all files.
    """
    all_docs: list[Document] = []
    for path in paths:
        docs = load_document(path)
        all_docs.extend(docs)
    return all_docs
