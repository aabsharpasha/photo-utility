# Liveness API - InsightFace + Anti-spoof ONNX (uv-based, Docker-first)
# Build: docker build -t liveness-api .  (anti-spoof model downloaded at build by default)
# Run:   docker run -p 8000:8000 liveness-api
ARG ANTISPOOF_MODEL_URL=https://github.com/SuriAI/face-antispoof-onnx/releases/download/v1.0.0/best_model.onnx
FROM python:3.11-slim AS builder
WORKDIR /build
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

# Install uv and build deps for insightface (Cython/C++ extensions)
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Install deps with uv (no venv in builder; we copy into runtime)
COPY pyproject.toml .
RUN uv sync --no-dev --no-install-project

# Download InsightFace buffalo_l at build time (so container start has no download)
# InsightFace uses ~/.insightface by default; set HOME so it writes under /build
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
  && rm -rf /var/lib/apt/lists/* \
  && HOME=/build .venv/bin/python -c "\
from insightface.app import FaceAnalysis; \
a = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider']); \
a.prepare(ctx_id=-1, det_size=(640, 640))\
"

# Download anti-spoof ONNX at build time (SuriAI MiniFAS v1.0.0). Override with build-arg to use another URL.
ARG ANTISPOOF_MODEL_URL=https://github.com/SuriAI/face-antispoof-onnx/releases/download/v1.0.0/best_model.onnx
RUN mkdir -p /build/models \
  && apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/* \
  && curl -fsSL "$ANTISPOOF_MODEL_URL" -o /build/models/antispoof.onnx

# Runtime stage
FROM python:3.11-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
COPY pyproject.toml .
COPY --from=builder /build/.venv /app/.venv
COPY --from=builder /build/models /app/models
COPY --from=builder /build/.insightface /app/.insightface
COPY app/ ./app/
# App tuning defaults live in app/config.py; optional host .env is merged by Pydantic when present.

# So InsightFace uses /app/.insightface (not /root/.insightface) and finds baked-in models
ENV HOME=/app
ENV INSIGHTFACE_HOME=/app/.insightface
EXPOSE 8000

# Run via venv Python so uvicorn is found regardless of venv layout/symlinks
# No Docker HEALTHCHECK: platform (Railway, k8s, etc.) can probe /api/health if desired.
CMD ["/bin/sh", "-c", "exec /app/.venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
