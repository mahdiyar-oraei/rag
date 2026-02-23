"""Streamlit UI for NotebookLM-style RAG chat."""

import tempfile
from pathlib import Path

import streamlit as st

from src.config import ALLOWED_EXTENSIONS, HUBSPOT_ACCESS_TOKEN, OPENAI_API_KEY
from src.hubspot_loader import HubSpotLoader
from src.ingestion import index_documents, ingest_documents, load_vectorstore
from src.retrieval import correct_query, create_rag_chain, get_retriever


def init_session_state():
    """Initialize Streamlit session state."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "rag_chain" not in st.session_state:
        st.session_state.rag_chain = None


def main():
    st.set_page_config(page_title="NotebookLM-style RAG", page_icon="📚")
    init_session_state()

    if not OPENAI_API_KEY:
        st.error(
            "OpenAI API key not found. Copy `.env.example` to `.env` and add your key."
        )
        st.stop()

    # Sidebar: document upload and indexing
    with st.sidebar:
        st.header("Documents")
        uploaded_files = st.file_uploader(
            "Upload PDF, Markdown, or Text files",
            type=["pdf", "md", "txt"],
            accept_multiple_files=True,
        )

        if st.button("Index documents", type="primary") and uploaded_files:
            paths = []
            with st.spinner("Indexing..."):
                try:
                    for f in uploaded_files:
                        suffix = Path(f.name).suffix.lower()
                        if suffix not in ALLOWED_EXTENSIONS:
                            st.warning(f"Skipping unsupported file: {f.name}")
                            continue
                        with tempfile.NamedTemporaryFile(
                            delete=False, suffix=suffix
                        ) as tmp:
                            tmp.write(f.read())
                            paths.append(tmp.name)

                    if paths:
                        vectorstore = index_documents(paths)
                        retriever = get_retriever(vectorstore)
                        st.session_state.rag_chain = create_rag_chain(retriever)
                        st.success(f"Indexed {len(paths)} file(s).")
                    else:
                        st.warning("No valid files to index.")
                except Exception as e:
                    st.error(f"Indexing failed: {e}")
                finally:
                    for p in paths:
                        Path(p).unlink(missing_ok=True)

        # Try to load existing vectorstore on startup
        if st.session_state.rag_chain is None:
            try:
                vectorstore = load_vectorstore()
                if vectorstore:
                    retriever = get_retriever(vectorstore)
                    st.session_state.rag_chain = create_rag_chain(retriever)
                    st.sidebar.success("Loaded existing index.")
            except Exception:
                pass

        st.divider()

        # HubSpot sync
        st.header("HubSpot CRM")
        if not HUBSPOT_ACCESS_TOKEN:
            st.warning("Set `HUBSPOT_ACCESS_TOKEN` in your `.env` file to enable sync.")
        else:
            if st.button("Sync from HubSpot", type="secondary"):
                try:
                    loader = HubSpotLoader()

                    def on_progress(entity: str, count: int) -> None:
                        progress_placeholder.markdown(
                            f"↳ **{entity}**: **{count:,}** fetched"
                        )

                    with st.status("Syncing from HubSpot...", expanded=True) as status:
                        progress_placeholder = st.empty()
                        progress_placeholder.markdown("_Waiting for first page…_")
                        st.write("Fetching contacts...")
                        contacts = loader.load_contacts(on_progress=on_progress)
                        st.write(f"✓ Fetched {len(contacts):,} contacts")

                        st.write("Fetching companies...")
                        companies = loader.load_companies(on_progress=on_progress)
                        st.write(f"✓ Fetched {len(companies):,} companies")

                        st.write("Fetching deals...")
                        deals = loader.load_deals(on_progress=on_progress, companies=companies)
                        st.write(f"✓ Fetched {len(deals):,} deals")

                        st.write("Fetching owners...")
                        owners = loader.load_owners(on_progress=on_progress)
                        st.write(f"✓ Fetched {len(owners):,} owners")

                        docs = contacts + companies + deals + owners
                        if docs:
                            st.write("Embedding and indexing...")
                            vectorstore = ingest_documents(docs)
                            retriever = get_retriever(vectorstore)
                            st.session_state.rag_chain = create_rag_chain(retriever)
                            st.write("✓ Indexed into vector store")
                            status.update(
                                label="Sync complete",
                                state="complete",
                                expanded=False,
                            )
                            st.success(
                                f"Synced {len(docs)} records: "
                                + ", ".join(
                                    f"{k}={v}"
                                    for k, v in [
                                        ("contacts", len(contacts)),
                                        ("companies", len(companies)),
                                        ("deals", len(deals)),
                                        ("owners", len(owners)),
                                    ]
                                )
                            )
                        else:
                            status.update(
                                label="No records found",
                                state="complete",
                                expanded=False,
                            )
                            st.warning("No CRM records found in HubSpot.")
                except Exception as e:
                    st.error(f"HubSpot sync failed: {e}")

        st.divider()
        if st.session_state.rag_chain:
            st.caption("Ready to chat. Ask questions about your documents.")
        else:
            st.caption("Upload and index documents to get started.")

    # Main chat area
    st.title("📚 NotebookLM-style RAG")
    st.caption("Ask questions about your indexed documents.")

    if st.session_state.rag_chain is None:
        st.info(
            "Upload documents in the sidebar and click **Index documents** to begin."
        )
        return

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask a question about your documents"):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    corrected = correct_query(prompt)
                    if corrected.lower() != prompt.lower():
                        st.caption(f"Interpreted as: _{corrected}_")
                    response = st.session_state.rag_chain.invoke({"input": corrected})
                    answer = response.get("answer", "No answer generated.")
                    st.markdown(answer)
                except Exception as e:
                    st.error(f"Error: {e}")
                    answer = str(e)

        st.session_state.messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
