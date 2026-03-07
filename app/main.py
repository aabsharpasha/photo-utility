"""
FastAPI application: enterprise liveness check API using InsightFace (Direct).

- Health / readiness probes
- Rate limiting
- Request size limit
- OpenAPI docs at /docs and /redoc
"""

import time
import uuid

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from app import __version__
from app.api.routes import router
from app.config import get_settings
from app.deps import limiter
from app.logging_config import configure_logging, get_logger

configure_logging()
logger = get_logger(__name__)

settings = get_settings()
app = FastAPI(
    title=settings.app_name,
    description="Face liveness check using InsightFace (Direct) for detection and heuristics.",
    version=__version__,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_logging(request: Request, call_next):
    """Add request_id and log duration."""
    request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "%s %s %s %.2fms",
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
        extra={"request_id": request_id, "duration_ms": round(duration_ms, 2)},
    )
    response.headers["x-request-id"] = request_id
    return response


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Return 500 with safe message in production."""
    logger.exception("Unhandled error: %s", exc)
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error" if settings.environment == "production" else str(exc),
        },
    )


app.include_router(router, prefix="/api")
