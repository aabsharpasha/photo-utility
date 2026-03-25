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
from app.services.replay_guard import motion_sequence_replay_metrics

logger = get_logger(__name__)
router = APIRouter()


def _motion_live_gate_summary(
    *,
    frames_live_ok: bool,
    motion_ok: bool,
    identity_ok: bool,
    single_face_ok: bool,
    challenge_ok: bool,
    moire_ok: bool,
    moire_gate_enabled: bool,
) -> dict[str, list[str]]:
    """Which motion liveness sub-checks passed vs failed (live requires all)."""
    passed: list[str] = []
    failed: list[str] = []
    if frames_live_ok:
        passed.append("per_frame_liveness")
    else:
        failed.append("per_frame_liveness")
    if motion_ok:
        passed.append("head_motion")
    else:
        failed.append("head_motion")
    if identity_ok:
        passed.append("face_identity_continuity")
    else:
        failed.append("face_identity_continuity")
    if single_face_ok:
        passed.append("single_face_per_frame")
    else:
        failed.append("single_face_per_frame")
    if challenge_ok:
        passed.append("challenge_response")
    else:
        failed.append("challenge_response")
    if moire_gate_enabled:
        if moire_ok:
            passed.append("moire_gate")
        else:
            failed.append("moire_gate")
    return {"passed": passed, "failed": failed}


def _motion_live_mismatch_explanation(
    *,
    frames_live_ok: bool,
    motion_ok: bool,
    identity_ok: bool,
    single_face_ok: bool,
    challenge_ok: bool,
    moire_ok: bool,
    moire_gate_enabled: bool,
    moire_max: float,
    moire_threshold: float,
    frames_live_count: int | None = None,
    frames_relaxed_live_count: int | None = None,
    n_frames: int | None = None,
    min_live_needed: int | None = None,
    relaxed_quorum_enabled: bool | None = None,
    strict_conf_thr: float | None = None,
    relaxed_conf_min: float | None = None,
) -> str | None:
    """Human-readable why live is false even when some scores look strong."""
    if (
        frames_live_ok
        and motion_ok
        and identity_ok
        and single_face_ok
        and challenge_ok
        and (not moire_gate_enabled or moire_ok)
    ):
        return None
    sentences: list[str] = []
    if not frames_live_ok:
        if (
            frames_live_count is not None
            and frames_relaxed_live_count is not None
            and n_frames is not None
            and min_live_needed is not None
            and relaxed_quorum_enabled is not None
            and strict_conf_thr is not None
            and relaxed_conf_min is not None
        ):
            sentences.append(
                "Per-frame liveness failed: need enough frames passing strict live "
                f"(confidence>={strict_conf_thr:.2f}) or, when enabled, relaxed quorum "
                f"(geometry + anti-spoof OK and confidence>={relaxed_conf_min:.2f}). "
                f"Counts: strict_live_frames={frames_live_count}/{n_frames}, "
                f"relaxed_eligible_frames={frames_relaxed_live_count}/{n_frames}, "
                f"need>={min_live_needed}, relaxed_quorum_enabled={relaxed_quorum_enabled}."
            )
        else:
            sentences.append(
                "Per-frame liveness failed: not enough frames pass face / anti-spoof / confidence gates."
            )
    if not motion_ok:
        sentences.append(
            "Head motion failed: consecutive face-center movement is below the configured minimum."
        )
    if not identity_ok:
        sentences.append(
            "Identity continuity failed: embedding similarity between some consecutive frames is too low."
        )
    if not single_face_ok:
        sentences.append(
            "Single-face gate failed: each frame must contain exactly one face."
        )
    if not challenge_ok:
        sentences.append(
            "Challenge-response failed: observed head-direction sequence does not match the requested pattern."
        )
    if moire_gate_enabled and not moire_ok:
        sentences.append(
            f"Moiré gate failed (screen-like periodic pattern heuristic): moire_max={moire_max:.4f} "
            f"must be strictly below {moire_threshold:.4f}. Per-frame liveness and motion/identity can still pass."
        )
    return " ".join(sentences) if sentences else None


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
    description=(
        "Submit a single image (base64). InsightFace detection plus anti-spoof (face + context crop when "
        "antispoof_dual_crop is enabled) and sharpness/size heuristics. Intended for real capture, not a "
        "screengrab or printed photo."
    ),
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
        "Multi-frame liveness tuned to reject presentation attacks (screen/video replay, held prints): "
        "each frame must pass face + context anti-spoof, optional moiré gate for display patterns, "
        "consecutive pairs must show movement, and (by default) the same face identity across frames. "
        "Tuning: app/config.py (motion_antispoof_*, motion_moire_*, motion_require_all_frames_live)."
    ),
)
async def liveness_motion(
    request: Request,
    body: MotionLivenessRequest,
    service: LivenessService = Depends(get_liveness_service),
    face_match: FaceMatchService = Depends(get_face_match_service),
    _api_key: str = Depends(require_api_key_query),
) -> LivenessResponse:
    """Motion-based liveness: multiple frames, per-pair motion, optional same-identity check."""
    settings = get_settings()
    if len(body.frames) < settings.motion_min_frames:
        raise HTTPException(
            status_code=400,
            detail=f"At least {settings.motion_min_frames} frames are required for motion liveness",
        )

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
    motion_real_thr = settings.motion_antispoof_real_threshold
    motion_ctx_thr = settings.motion_antispoof_context_real_threshold
    motion_logit_min = settings.motion_antispoof_min_logit_diff
    for img in cv_frames:
        res = service.check(
            img,
            force_context_antispoof=settings.motion_require_context_antispoof,
            context_antispoof_enabled_override=settings.motion_require_context_antispoof,
            face_area_min_ratio_override=settings.motion_face_area_min_ratio,
            laplacian_min_override=settings.motion_laplacian_min,
            antispoof_real_threshold_override=motion_real_thr,
            antispoof_context_real_threshold_override=motion_ctx_thr,
            antispoof_min_logit_diff_override=motion_logit_min,
            liveness_confidence_threshold_override=settings.motion_frame_liveness_confidence_threshold,
        )
        frame_results.append(res)
        bbox = res.details.get("bbox")
        h, w = img.shape[:2]
        if bbox and len(bbox) == 4:
            x, y, bw, bh = bbox
            cx = x + bw / 2.0
            cy = y + bh / 2.0
            centers.append((cx, cy))
            norms.append(max(h, w))

    import math

    motion_pair_shifts: list[float] = []
    motion_max_shift = 0.0
    if len(centers) >= 2:
        for i in range(len(centers) - 1):
            (cx1, cy1), (cx2, cy2) = centers[i], centers[i + 1]
            dx = cx2 - cx1
            dy = cy2 - cy1
            norm = max(norms[i], norms[i + 1]) or 1.0
            shift = math.hypot(dx, dy) / norm
            motion_pair_shifts.append(float(round(shift, 4)))
            motion_max_shift = max(motion_max_shift, shift)

    min_shift = float(settings.motion_min_normalized_shift)
    if not motion_pair_shifts:
        motion_ok = False
    elif settings.motion_require_shift_all_consecutive_pairs:
        motion_ok = all(s >= min_shift for s in motion_pair_shifts)
    else:
        motion_ok = motion_max_shift >= min_shift

    # Enforce server-side default challenge for all clients.
    # This keeps behavior consistent even if callers pass another challenge value.
    challenge = "side-to-side"
    challenge_ok = True
    challenge_details: dict[str, object] = {"challenge": challenge}
    if challenge != "none":
        if len(centers) < 3:
            challenge_ok = False
            challenge_details["reason"] = "not_enough_detected_faces_for_challenge"
        else:
            dx1 = centers[1][0] - centers[0][0]
            dx2 = centers[2][0] - centers[1][0]
            min_turn_shift = min_shift * 0.6
            if challenge == "left-right":
                challenge_ok = (dx1 <= -min_turn_shift) and (dx2 >= min_turn_shift)
            elif challenge == "right-left":
                challenge_ok = (dx1 >= min_turn_shift) and (dx2 <= -min_turn_shift)
            challenge_details.update(
                {
                    "dx1": float(round(dx1, 4)),
                    "dx2": float(round(dx2, 4)),
                    "min_turn_shift": float(round(min_turn_shift, 4)),
                }
            )

    identity_ok = True
    consecutive_similarities: list[float] = []
    if settings.motion_face_consistency_enabled and len(cv_frames) >= 2:
        thr = float(settings.motion_min_consecutive_face_similarity)
        for i in range(len(cv_frames) - 1):
            sim = face_match.pairwise_face_similarity_percent(cv_frames[i], cv_frames[i + 1])
            consecutive_similarities.append(float(round(sim, 2)))
            if sim < thr:
                identity_ok = False

    per_frame_face_counts = [int(r.details.get("face_count") or 0) for r in frame_results]
    single_face_ok = all(fc == 1 for fc in per_frame_face_counts) if per_frame_face_counts else False

    frames_live_count = sum(1 for r in frame_results if r.live)
    relaxed_min = float(settings.motion_relaxed_confidence_min)
    frames_relaxed_live_count = sum(
        1
        for r in frame_results
        if r.details.get("frame_geometry_ok") is True
        and r.details.get("antispoof_spoof_gate_ok") is True
        and float(r.confidence) >= relaxed_min
    )
    n_frames = len(frame_results)
    min_live_needed = min(settings.motion_min_frames_with_live_face, n_frames) if n_frames else 0
    if settings.motion_require_all_frames_live:
        frames_live_ok = n_frames > 0 and frames_live_count == n_frames
    else:
        strict_ok = n_frames > 0 and frames_live_count >= min_live_needed
        relaxed_ok = (
            bool(settings.motion_relaxed_frame_quorum_enabled)
            and n_frames > 0
            and frames_relaxed_live_count >= min_live_needed
        )
        frames_live_ok = strict_ok or relaxed_ok
    used_relaxed_frame_quorum = (
        bool(settings.motion_relaxed_frame_quorum_enabled)
        and not settings.motion_require_all_frames_live
        and frames_live_ok
        and frames_live_count < min_live_needed
        and frames_relaxed_live_count >= min_live_needed
    )

    confs = [float(r.confidence) for r in frame_results]
    min_conf = min(confs) if confs else 0.0
    mean_conf = sum(confs) / len(confs) if confs else 0.0
    # One motion-blurred frame should not zero top-level confidence if the sequence passes.
    aggregate_confidence = min(1.0, 0.35 * min_conf + 0.65 * mean_conf)

    replay_metrics = motion_sequence_replay_metrics(cv_frames)
    moire_ok = True
    if settings.motion_moire_gate_enabled:
        moire_ok = replay_metrics["moire_max"] < settings.motion_moire_max_score

    live_candidate = bool(
        frames_live_ok and motion_ok and moire_ok and identity_ok and single_face_ok and challenge_ok
    )
    rejection_reasons: list[str] = []
    if not frames_live_ok:
        rejection_reasons.append("per_frame_liveness")
    if not motion_ok:
        rejection_reasons.append("insufficient_head_motion")
    if not identity_ok:
        rejection_reasons.append("face_identity_mismatch")
    if not single_face_ok:
        rejection_reasons.append("multiple_or_missing_faces")
    if not challenge_ok:
        rejection_reasons.append(f"challenge_response_failed({challenge})")
    if settings.motion_moire_gate_enabled and not moire_ok:
        rejection_reasons.append(
            f"moire_gate(moire_max={replay_metrics['moire_max']:.4f}>="
            f"{float(settings.motion_moire_max_score):.4f})"
        )

    gate_summary = _motion_live_gate_summary(
        frames_live_ok=frames_live_ok,
        motion_ok=motion_ok,
        identity_ok=identity_ok,
        single_face_ok=single_face_ok,
        challenge_ok=challenge_ok,
        moire_ok=moire_ok,
        moire_gate_enabled=settings.motion_moire_gate_enabled,
    )
    mismatch_text = _motion_live_mismatch_explanation(
        frames_live_ok=frames_live_ok,
        motion_ok=motion_ok,
        identity_ok=identity_ok,
        single_face_ok=single_face_ok,
        challenge_ok=challenge_ok,
        moire_ok=moire_ok,
        moire_gate_enabled=settings.motion_moire_gate_enabled,
        moire_max=float(replay_metrics["moire_max"]),
        moire_threshold=float(settings.motion_moire_max_score),
        frames_live_count=frames_live_count,
        frames_relaxed_live_count=frames_relaxed_live_count,
        n_frames=n_frames,
        min_live_needed=min_live_needed,
        relaxed_quorum_enabled=bool(settings.motion_relaxed_frame_quorum_enabled),
        strict_conf_thr=float(settings.motion_frame_liveness_confidence_threshold),
        relaxed_conf_min=relaxed_min,
    )

    details = {
        "frame_count": len(frame_results),
        "motion_min_frames": settings.motion_min_frames,
        "motion_ok": motion_ok,
        "motion_pair_shift_ratios": motion_pair_shifts,
        "motion_max_shift_ratio": float(round(motion_max_shift, 4)),
        "motion_min_normalized_shift": min_shift,
        "motion_require_all_pairs": settings.motion_require_shift_all_consecutive_pairs,
        "challenge": challenge,
        "challenge_ok": challenge_ok,
        "challenge_details": challenge_details,
        "identity_ok": identity_ok,
        "single_face_ok": single_face_ok,
        "per_frame_face_counts": per_frame_face_counts,
        "motion_face_consistency_enabled": settings.motion_face_consistency_enabled,
        "consecutive_face_similarities": consecutive_similarities,
        "motion_min_consecutive_face_similarity": float(settings.motion_min_consecutive_face_similarity),
        "motion_face_area_min_ratio": float(settings.motion_face_area_min_ratio),
        "motion_laplacian_min": float(settings.motion_laplacian_min),
        "motion_require_context_antispoof": settings.motion_require_context_antispoof,
        "motion_frame_liveness_confidence_threshold": float(
            settings.motion_frame_liveness_confidence_threshold
        ),
        "motion_relaxed_frame_quorum_enabled": bool(settings.motion_relaxed_frame_quorum_enabled),
        "motion_relaxed_confidence_min": relaxed_min,
        "frames_live_count": frames_live_count,
        "frames_relaxed_live_count": frames_relaxed_live_count,
        "motion_used_relaxed_frame_quorum": used_relaxed_frame_quorum,
        "motion_require_all_frames_live": settings.motion_require_all_frames_live,
        "motion_min_frames_with_live_face": min_live_needed,
        "aggregate_confidence": float(round(aggregate_confidence, 4)),
        "replay_metrics": replay_metrics,
        "moire_gate_enabled": settings.motion_moire_gate_enabled,
        "moire_gate_ok": moire_ok,
        "per_frame": [
            {
                "live": r.live,
                "confidence": float(r.confidence),
                "details": {
                    "face_count": r.details.get("face_count"),
                    "bbox": r.details.get("bbox"),
                    "reason": r.details.get("reason"),
                    "antispoof_real_score": r.details.get("antispoof_real_score"),
                    "antispoof_logit_diff": r.details.get("antispoof_logit_diff"),
                    "antispoof_context_real_score": r.details.get("antispoof_context_real_score"),
                    "antispoof_context_logit_diff": r.details.get("antispoof_context_logit_diff"),
                    "antispoof_face_gate_ok": r.details.get("antispoof_face_gate_ok"),
                    "antispoof_context_gate_ok": r.details.get("antispoof_context_gate_ok"),
                    "antispoof_dual_crop_strategy": r.details.get("antispoof_dual_crop_strategy"),
                    "frame_geometry_ok": r.details.get("frame_geometry_ok"),
                    "antispoof_spoof_gate_ok": r.details.get("antispoof_spoof_gate_ok"),
                },
            }
            for r in frame_results
        ],
        "live_rejection_reasons": rejection_reasons,
        "live_gate_summary": gate_summary,
        "live_mismatch_explanation": mismatch_text,
    }

    live = live_candidate
    # Report real blended score even when live is false (e.g. moiré gate only) so clients see the mismatch.
    confidence = float(round(aggregate_confidence, 4))

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


