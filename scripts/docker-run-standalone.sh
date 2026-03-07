#!/usr/bin/env sh
# Build and run with plain Docker (no Compose, no buildx required).
# Use if "docker compose build" gives Bake/buildx warnings or fails.
set -e
cd "$(dirname "$0")/.."
docker build -t liveness-api:latest .
docker volume create insightface-cache 2>/dev/null || true
docker run --rm -p 8000:8000 \
  -e ENVIRONMENT=production \
  -e LOG_LEVEL=INFO \
  -v insightface-cache:/app/.insightface \
  --name liveness-api \
  liveness-api:latest
