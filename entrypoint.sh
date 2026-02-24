#!/bin/bash
set -e

mkdir -p /data/chroma_db

export PORT="${PORT:-8080}"
envsubst '${PORT}' < /app/nginx.conf.template > /etc/nginx/nginx.conf

exec supervisord -n -c /app/supervisord.conf
