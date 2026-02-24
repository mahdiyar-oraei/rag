#!/usr/bin/env bash
# Start both the Facebook Messenger webhook (FastAPI) and the Streamlit admin panel.
set -e
cd "$(dirname "$0")"
trap 'kill $uvicorn_pid $streamlit_pid 2>/dev/null' EXIT
uvicorn webhook:app --host 0.0.0.0 --port 8000 &
uvicorn_pid=$!
streamlit run app.py --server.port 8501 &
streamlit_pid=$!
echo "Webhook: http://localhost:8000 (PID $uvicorn_pid)"
echo "Admin:   http://localhost:8501 (PID $streamlit_pid)"
wait
