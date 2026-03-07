"""API route handlers."""

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse

from app import __version__
from app.api.schemas import LivenessRequest, LivenessResponse
from app.config import get_settings
from app.logging_config import get_logger, log_extra
from app.services.liveness import LivenessService, decode_image, get_liveness_service

logger = get_logger(__name__)
router = APIRouter()


@router.get("/health", tags=["health"])
async def health(request: Request) -> dict:
    """Liveness probe: fast, no model load."""
    settings = get_settings()
    return {
        "status": "ok",
        "version": __version__,
        "environment": settings.environment,
    }


@router.get("/ready", tags=["health"])
async def ready(request: Request) -> dict:
    """Readiness: optionally trigger model load so orchestrator knows we're ready."""
    try:
        from app.services.liveness import _get_face_app
        _get_face_app()
    except Exception as e:
        logger.warning("Readiness check failed: %s", e)
        return JSONResponse(
            status_code=503,
            content={
                "status": "degraded",
                "reason": str(e),
                "version": __version__,
            },
        )
    settings = get_settings()
    return {
        "status": "ok",
        "version": __version__,
        "model": settings.insightface_model,
    }


@router.post(
    "/v1/liveness",
    response_model=LivenessResponse,
    tags=["liveness"],
    summary="Check face liveness",
    description="Submit a single image (base64). Uses InsightFace for detection and sharpness/size heuristics for liveness.",
)
async def liveness(
    request: Request,
    body: LivenessRequest,
    service: LivenessService = Depends(get_liveness_service),
) -> LivenessResponse:
    """Run liveness check on the provided image."""
    settings = get_settings()
    if len(body.image_base64.encode("utf-8")) > settings.max_image_size_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"Image payload exceeds max size ({settings.max_image_size_bytes} bytes)",
        )
    cv_img = decode_image(body.image_base64)
    if cv_img is None:
        raise HTTPException(status_code=400, detail="Invalid or unsupported image_base64")
    result = service.check(cv_img)
    logger.info(
        "liveness check",
        extra=log_extra(
            live=result.live,
            confidence=result.confidence,
            face_count=result.details.get("face_count"),
        ),
    )
    return LivenessResponse(**result.to_dict())


# Alias for convenience
@router.post(
    "/liveness",
    response_model=LivenessResponse,
    tags=["liveness"],
    include_in_schema=False,
)
async def liveness_legacy(
    request: Request,
    body: LivenessRequest,
    service: LivenessService = Depends(get_liveness_service),
) -> LivenessResponse:
    """Legacy path: same as POST /v1/liveness."""
    return await liveness(request, body, service)
