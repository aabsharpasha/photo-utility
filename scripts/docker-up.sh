#!/usr/bin/env sh
# Build and run Liveness API with Docker.
# Use this if "docker compose up --build" fails on your Docker version.
set -e
cd "$(dirname "$0")/.."
docker compose build
docker compose up
