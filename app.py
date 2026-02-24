"""Streamlit UI for NotebookLM-style RAG chat."""

import tempfile
from pathlib import Path

import streamlit as st

from src.config import ALLOWED_EXTENSIONS, HUBSPOT_ACCESS_TOKEN, OPENAI_API_KEY
from src.db import get_all_linked, get_unlinked_psids, init_db, link_psid_to_contact
from src.hubspot_cache import get_cache_counts, get_cache_timestamp, load_hubspot_docs
from src.hubspot_loader import HubSpotLoader
from src.ingestion import index_documents, ingest_documents, load_vectorstore
from src.retrieval import correct_query, create_rag_chain, get_retriever

init_db()


def init_session_state():
    """Initialize Streamlit session state."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "rag_chain" not in st.session_state:
        st.session_state.rag_chain = None
    if "fb_contacts" not in st.session_state:
        st.session_state.fb_contacts = []


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
                        contacts = loader.load_contacts(
                            on_progress=on_progress,
                            force_refresh=True,
                        )
                        st.write(f"✓ Fetched {len(contacts):,} contacts")

                        st.write("Fetching companies...")
                        companies = loader.load_companies(
                            on_progress=on_progress,
                            force_refresh=True,
                        )
                        st.write(f"✓ Fetched {len(companies):,} companies")

                        st.write("Fetching deals...")
                        deals = loader.load_deals(
                            on_progress=on_progress,
                            companies=companies,
                            force_refresh=True,
                        )
                        st.write(f"✓ Fetched {len(deals):,} deals")

                        st.write("Fetching owners...")
                        owners = loader.load_owners(
                            on_progress=on_progress,
                            force_refresh=True,
                        )
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

        # Cached data summary + index-from-DB button
        st.subheader("Cached data")
        counts = get_cache_counts()
        if counts:
            ts = get_cache_timestamp()
            if ts:
                st.caption(f"Last synced: {ts.strftime('%Y-%m-%d %H:%M')} UTC")
            _LABELS = {
                "contact": "Contacts",
                "company": "Companies",
                "deal": "Deals",
                "owner": "Owners",
            }
            cols = st.columns(2)
            for i, key in enumerate(["contact", "company", "deal", "owner"]):
                cols[i % 2].metric(_LABELS[key], f"{counts.get(key, 0):,}")
            if st.button("Index from database", type="primary"):
                try:
                    with st.status("Indexing cached data...", expanded=True) as status:
                        docs = load_hubspot_docs()
                        if not docs:
                            status.update(label="Cache is empty", state="error")
                            st.warning("No records in cache. Run **Sync from HubSpot** first.")
                        else:
                            st.write(f"Loading {len(docs):,} records from database...")
                            vectorstore = ingest_documents(docs)
                            retriever = get_retriever(vectorstore)
                            st.session_state.rag_chain = create_rag_chain(retriever)
                            st.write("✓ Embedded and indexed")
                            status.update(
                                label=f"Indexed {len(docs):,} records",
                                state="complete",
                                expanded=False,
                            )
                            st.success(
                                "Ready to chat using cached CRM data. "
                                + ", ".join(
                                    f"{_LABELS.get(k, k)}={v:,}"
                                    for k, v in sorted(counts.items())
                                )
                            )
                except Exception as e:
                    st.error(f"Indexing failed: {e}")
        else:
            st.caption("No data cached yet. Run **Sync from HubSpot** first.")

        st.divider()
        if st.session_state.rag_chain:
            st.caption("Ready to chat. Ask questions about your documents.")
        else:
            st.caption("Upload and index documents to get started.")

    # Main area: Chat and Facebook Connections tabs
    tab_chat, tab_fb = st.tabs(["Chat", "Facebook Connections"])

    with tab_chat:
        _render_chat_tab()

    with tab_fb:
        _render_facebook_tab()


def _render_chat_tab():
    """Render the main RAG chat tab."""
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


def _render_facebook_tab():
    """Render the Facebook Connections admin tab."""
    st.title("Facebook Messenger Connections")
    st.caption("Link Facebook users to HubSpot contacts so they can access their data via chat.")

    if not HUBSPOT_ACCESS_TOKEN:
        st.warning("Set `HUBSPOT_ACCESS_TOKEN` in your `.env` file to load contacts for linking.")
    else:
        col_load, col_refresh = st.columns(2)
        with col_load:
            if st.button("Load contacts from HubSpot", key="fb_load_contacts"):
                with st.spinner("Loading contacts (from cache if available)..."):
                    try:
                        loader = HubSpotLoader()
                        docs = loader.load_contacts(use_cache=True)
                        st.session_state.fb_contacts = [
                            {
                                "id": str(d.metadata.get("hs_object_id", "")),
                                "name": d.page_content.split("\n")[0].replace("Contact: ", "").strip() or "Unknown",
                                "email": next(
                                    (line.replace("Email: ", "").strip() for line in d.page_content.split("\n") if line.startswith("Email: ")),
                                    "N/A",
                                ),
                            }
                            for d in docs
                        ]
                        st.success(f"Loaded {len(st.session_state.fb_contacts)} contacts.")
                    except Exception as e:
                        st.error(f"Failed to load contacts: {e}")
        with col_refresh:
            if st.button("Refresh from API", key="fb_refresh_contacts"):
                with st.spinner("Refreshing contacts from HubSpot API..."):
                    try:
                        loader = HubSpotLoader()
                        docs = loader.load_contacts(use_cache=True, force_refresh=True)
                        st.session_state.fb_contacts = [
                            {
                                "id": str(d.metadata.get("hs_object_id", "")),
                                "name": d.page_content.split("\n")[0].replace("Contact: ", "").strip() or "Unknown",
                                "email": next(
                                    (line.replace("Email: ", "").strip() for line in d.page_content.split("\n") if line.startswith("Email: ")),
                                    "N/A",
                                ),
                            }
                            for d in docs
                        ]
                        st.success(f"Refreshed {len(st.session_state.fb_contacts)} contacts.")
                    except Exception as e:
                        st.error(f"Failed to refresh contacts: {e}")

    st.divider()

    st.subheader("Unmatched conversations")
    unlinked = get_unlinked_psids()
    if not unlinked:
        st.info("No unmatched conversations. New Facebook messages will appear here.")
    else:
        for row in unlinked:
            psid = row["psid"]
            with st.expander(f"PSID: {psid} — {row.get('message_count', 0)} message(s)"):
                preview = (row.get("message_preview") or "")[:200]
                if preview:
                    st.caption(f"Last message: {preview}")
                if st.session_state.fb_contacts:
                    search = st.text_input("Search contact", key=f"fb_search_{psid}", placeholder="Name or email...")
                    contacts = st.session_state.fb_contacts
                    if search:
                        q = search.lower()
                        contacts = [c for c in contacts if q in (c.get("name", "") + c.get("email", "")).lower()]
                    if contacts:
                        idx = st.selectbox(
                            "Select contact to link",
                            range(len(contacts)),
                            format_func=lambda i: f"{contacts[i]['name']} ({contacts[i]['email']})",
                            key=f"fb_select_{psid}",
                        )
                        if st.button("Link", key=f"fb_link_{psid}"):
                            c = contacts[idx]
                            link_psid_to_contact(psid, c["id"], c["name"])
                            st.success(f"Linked to {c['name']}.")
                            st.rerun()
                else:
                    st.caption("Click **Load contacts from HubSpot** above first.")

    st.divider()
    st.subheader("Matched conversations")
    linked = get_all_linked()
    if not linked:
        st.info("No linked conversations yet.")
    else:
        st.dataframe(
            linked,
            column_config={
                "psid": "Facebook PSID",
                "hubspot_contact_id": "HubSpot Contact ID",
                "contact_name": "Contact Name",
                "linked_at": "Linked At",
                "message_count": "Messages",
            },
            hide_index=True,
        )


if __name__ == "__main__":
    main()
