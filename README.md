# NotebookLM-style RAG Application

A Retrieval-Augmented Generation (RAG) application that lets you upload documents (PDF, Markdown, Text) and ask questions about them. Built with LangChain, ChromaDB, and OpenAI.

## Tech Stack

- **Orchestration**: LangChain
- **Embeddings**: OpenAI `text-embedding-3-small`
- **Vector DB**: ChromaDB (local, persistent)
- **LLM**: OpenAI `gpt-4o-mini`
- **UI**: Streamlit

## Setup

1. **Create a virtual environment and install dependencies**

   ```bash
   cd notebook
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Configure OpenAI API key**

   ```bash
   cp .env.example .env
   # Edit .env and add your OPENAI_API_KEY
   ```

3. **Run the app**

   ```bash
   streamlit run app.py
   ```

## Usage

1. Open the app in your browser (default: http://localhost:8501)
2. Upload PDF, Markdown, or Text files in the sidebar
3. Click **Index documents** to chunk, embed, and store them in ChromaDB
4. Ask questions in the chat — the RAG pipeline retrieves relevant chunks and generates answers

## Project Structure

```
notebook/
├── app.py              # Streamlit UI
├── src/
│   ├── config.py       # Settings (env vars, chunk params)
│   ├── loaders.py      # Document loaders (PDF, MD, TXT)
│   ├── ingestion.py   # Chunk, embed, ChromaDB storage
│   └── retrieval.py   # RAG chain (retriever + LLM)
├── chroma_db/         # Persisted vectors (created on first index)
└── data/              # Optional: place sample docs here
```

## Configuration

Environment variables (in `.env`):

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | (required) | Your OpenAI API key |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |
| `LLM_MODEL` | `gpt-4o-mini` | Chat model |
| `CHROMA_PERSIST_DIR` | `./chroma_db` | Vector store path |
| `CHUNK_SIZE` | `1000` | Tokens per chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `TOP_K` | `3` | Number of chunks to retrieve |

## License

MIT
