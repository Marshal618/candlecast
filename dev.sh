#!/usr/bin/env bash
# Run backend (FastAPI) and frontend (Vite :5173) together. Ctrl+C stops both.
# API_PORT=9000 ./dev.sh to change the backend port.
set -euo pipefail
export API_PORT="${API_PORT:-8010}"
ROOT="$(cd "$(dirname "$0")" && pwd)"

if [ ! -d "$ROOT/backend/.venv" ]; then
  echo "Setting up Python venv (needs Python 3.10–3.13)…"
  PY="$(command -v python3.12 || command -v python3.11 || command -v python3.10 || echo python3)"
  "$PY" -m venv "$ROOT/backend/.venv"
  "$ROOT/backend/.venv/bin/pip" install -r "$ROOT/backend/requirements.txt"
fi
if [ ! -d "$ROOT/frontend/node_modules" ]; then
  (cd "$ROOT/frontend" && npm install)
fi

trap 'kill 0' EXIT
(cd "$ROOT/backend" && .venv/bin/uvicorn app.main:app --host 127.0.0.1 --port "$API_PORT" --reload --reload-dir app) &
(cd "$ROOT/frontend" && npm run dev -- --host 127.0.0.1) &
wait
