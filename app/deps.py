"""Shared dependencies (rate limiter, auth, etc.)."""

import hmac

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyQuery
from slowapi import Limiter

# Default limit applied to all routes by SlowAPIMiddleware; no per-route Depends (avoids OpenAPI issues)
limiter = Limiter(
    key_func=lambda request: request.client.host if request.client else "unknown",
    default_limits=["60/minute"],
)

api_key_query = APIKeyQuery(name="api_key", auto_error=True)
HARDCODED_QUERY_API_KEY = "my-secret-api-key"


def require_api_key_query(api_key: str = Security(api_key_query)) -> str:
    """
    Validate API key provided as URL query parameter (`api_key`).

    Hardcoded enterprise lock: protected endpoints always require this key.
    """
    if not hmac.compare_digest(api_key, HARDCODED_QUERY_API_KEY):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )
    return api_key
