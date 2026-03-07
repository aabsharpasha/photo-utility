"""Shared dependencies (rate limiter, etc.)."""

from slowapi import Limiter

# Default limit applied to all routes by SlowAPIMiddleware; no per-route Depends (avoids OpenAPI issues)
limiter = Limiter(
    key_func=lambda request: request.client.host if request.client else "unknown",
    default_limits=["60/minute"],
)
