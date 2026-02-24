#!/bin/bash
set -e

mkdir -p /data/chroma_db

# Ensure DB parent dir exists (default: /data)
mkdir -p /data

# Verify webhook module loads before starting (surfaces import errors)
cd /app && python -c "from webhook import app" 2>&1 || { echo "Webhook import failed - see above"; exit 1; }

export PORT="${PORT:-8080}"
envsubst '${PORT}' < /app/nginx.conf.template > /etc/nginx/nginx.conf

exec supervisord -n -c /app/supervisord.conf
