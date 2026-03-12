"""API route handlers."""

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse

from app import __version__
from app.api.schemas import (
    CompareFacesRequest,
    CompareFacesResponse,
    LivenessRequest,
    LivenessResponse,
    MotionLivenessRequest,
)
from app.config import get_settings
from app.deps import require_api_key_query
from app.logging_config import get_logger, log_extra
from app.services.liveness import LivenessService, decode_image, get_liveness_service, _to_native
from app.services.face_match import FaceMatchService, get_face_match_service

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
    _api_key: str = Depends(require_api_key_query),
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


@router.post(
    "/v1/liveness-motion",
    response_model=LivenessResponse,
    tags=["liveness"],
    summary="Motion-based liveness (multiple frames)",
    description=(
        "Submit 2+ images captured in sequence. Each frame is checked for liveness, "
        "and basic motion (head movement) is required between frames."
    ),
)
async def liveness_motion(
    request: Request,
    body: MotionLivenessRequest,
    service: LivenessService = Depends(get_liveness_service),
    _api_key: str = Depends(require_api_key_query),
) -> LivenessResponse:
    """Motion-based liveness: require multiple frames and detectable motion."""
    settings = get_settings()
    if len(body.frames) < 2:
        raise HTTPException(status_code=400, detail="At least 2 frames are required for motion liveness")

    # Decode all frames
    cv_frames = []
    for idx, b64 in enumerate(body.frames):
        if len(b64.encode("utf-8")) > settings.max_image_size_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"Frame {idx} exceeds max size ({settings.max_image_size_bytes} bytes)",
            )
        img = decode_image(b64)
        if img is None:
            raise HTTPException(status_code=400, detail=f"Invalid or unsupported image_base64 in frame {idx}")
        cv_frames.append(img)

    # Per-frame liveness
    frame_results = []
    centers = []
    norms = []
    for img in cv_frames:
        res = service.check(img)
        frame_results.append(res)
        bbox = res.details.get("bbox")
        h, w = img.shape[:2]
        if bbox and len(bbox) == 4:
            x, y, bw, bh = bbox
            cx = x + bw / 2.0
            cy = y + bh / 2.0
            centers.append((cx, cy))
            norms.append(max(h, w))

    # Basic motion: max normalized center shift across frames
    import math

    motion_max_shift = 0.0
    if len(centers) >= 2:
        for i in range(len(centers) - 1):
            (cx1, cy1), (cx2, cy2) = centers[i], centers[i + 1]
            dx = cx2 - cx1
            dy = cy2 - cy1
            # normalize by max image size (use corresponding norm)
            norm = max(norms[i], norms[i + 1]) or 1.0
            shift = math.hypot(dx, dy) / norm
            motion_max_shift = max(motion_max_shift, shift)

    # Require some motion between frames (heuristic).
    # 0.01 = 1% of image size (more forgiving; real users only need slight head movement).
    motion_min_shift = 0.01
    motion_ok = motion_max_shift >= motion_min_shift

    all_live = all(r.live for r in frame_results)
    min_conf = min((r.confidence for r in frame_results), default=0.0)

    details = {
        "frame_count": len(frame_results),
        "motion_ok": motion_ok,
        "motion_max_shift_ratio": float(round(motion_max_shift, 4)),
        "per_frame": [
            {
                "live": r.live,
                "confidence": float(r.confidence),
                "details": {
                    "face_count": r.details.get("face_count"),
                    "bbox": r.details.get("bbox"),
                },
            }
            for r in frame_results
        ],
    }

    live = bool(all_live and motion_ok)
    confidence = float(min_conf if live else 0.0)

    logger.info(
        "liveness motion check",
        extra=log_extra(
            live=live,
            confidence=confidence,
            face_count=details["per_frame"][0]["details"]["face_count"] if details["per_frame"] else None,
        ),
    )

    return LivenessResponse(
        live=live,
        confidence=confidence,
        details=_to_native(details),
        errors=[],
    )


@router.post(
    "/v1/face-match",
    response_model=CompareFacesResponse,
    tags=["face-match"],
    summary="Compare faces (Rekognition-style)",
    description=(
        "Compare a source face against faces in a target image. "
        "Request and response formats follow AWS Rekognition CompareFaces (subset)."
    ),
)
async def face_match(
    request: Request,
    body: CompareFacesRequest,
    service: FaceMatchService = Depends(get_face_match_service),
    _api_key: str = Depends(require_api_key_query),
) -> CompareFacesResponse:
    """
    Face match endpoint using InsightFace embeddings but Rekognition-style schema.
    """
    settings = get_settings()

    # Enforce payload limits (Bytes are base64 strings)
    for label, img in (("SourceImage", body.SourceImage), ("TargetImage", body.TargetImage)):
        if len(img.Bytes.encode("utf-8")) > settings.max_image_size_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"{label} payload exceeds max size ({settings.max_image_size_bytes} bytes)",
            )

    src_img = decode_image(body.SourceImage.Bytes)
    tgt_img = decode_image(body.TargetImage.Bytes)
    if src_img is None:
        raise HTTPException(status_code=400, detail="Invalid or unsupported SourceImage.Bytes")
    if tgt_img is None:
        raise HTTPException(status_code=400, detail="Invalid or unsupported TargetImage.Bytes")

    similarity_threshold = body.SimilarityThreshold
    result_dict = service.compare(src_img, tgt_img, similarity_threshold=similarity_threshold)
    result_dict["Match"] = len(result_dict.get("FaceMatches") or []) > 0

    logger.info(
        "face match",
        extra=log_extra(
            similarity_threshold=similarity_threshold,
            source_face_present=bool(result_dict.get("SourceImageFace")),
            match_count=len(result_dict.get("FaceMatches") or []),
            match=result_dict["Match"],
        ),
    )

    return CompareFacesResponse(**result_dict)


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
    _api_key: str = Depends(require_api_key_query),
) -> LivenessResponse:
    """Legacy path: same as POST /v1/liveness."""
    return await liveness(request, body, service)
