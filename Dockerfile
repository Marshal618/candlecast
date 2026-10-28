# Single container: builds the frontend, then serves it + the API from FastAPI on :8000.
# CPU-only torch by default. For a GPU host, swap the base image for a CUDA one and
# drop the --index-url line so pip installs CUDA torch.

FROM node:22-slim AS frontend
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.12-slim
WORKDIR /app/backend
ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/app/hf-cache \
    KD_CACHE_PATH=/app/data/cache.sqlite3 \
    KD_JOURNAL_PATH=/app/data/journal.sqlite3 \
    KD_SIM_PATH=/app/data/sim.sqlite3 \
    KD_RESEARCH_PATH=/app/data/research.sqlite3 \
    KD_FRONTEND_DIST=/app/frontend/dist
COPY backend/requirements.txt ./
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt
COPY backend/ ./
COPY --from=frontend /app/frontend/dist /app/frontend/dist
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
