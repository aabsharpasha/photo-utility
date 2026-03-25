"""
Liveness check using InsightFace (Direct) + anti-spoof ONNX.

- Face detection: InsightFace buffalo_l (RetinaFace).
- Liveness: sharpness (Laplacian), face size, detection score + anti-spoof ONNX (MiniFAS-style).
"""

from __future__ import annotations

import base64
import os
import sys
from typing import Any

import cv2
import numpy as np


def _to_native(x: Any) -> Any:
    """Convert numpy scalars/arrays in structures to native Python for JSON serialization."""
    # Scalars
    if isinstance(x, (np.floating, np.integer)):
        return float(x) if isinstance(x, np.floating) else int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    # Arrays / containers
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, dict):
        return {k: _to_native(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_native(v) for v in x]
    return x

from app.config import get_settings
from app.logging_config import get_logger
from app.services.antispoof import run_antispoof

logger = get_logger(__name__)

# Lazy InsightFace app (heavy to load)
_face_app: Any = None


def _onnx_providers() -> list[str]:
    """Use CPU unless settings.use_gpu and CUDA provider is available."""
    if get_settings().use_gpu:
        try:
            import onnxruntime as ort
            available = set(ort.get_available_providers())
            if "CUDAExecutionProvider" in available:
                return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        except Exception:
            pass
    return ["CPUExecutionProvider"]


def _get_face_app():
    """Lazy-load InsightFace FaceAnalysis (singleton)."""
    global _face_app
    if _face_app is None:
        # Redirect fd 2 (stderr) so ONNX/InsightFace C++ output is suppressed (cpuid, "Applied providers", "model ignore" on Mac/Docker)
        _stderr_fd = sys.stderr.fileno() if hasattr(sys.stderr, "fileno") else 2
        _devnull = open(os.devnull, "w")
        _saved_fd = os.dup(_stderr_fd)
        try:
            os.dup2(_devnull.fileno(), _stderr_fd)
            from insightface.app import FaceAnalysis
            settings = get_settings()
            root = settings.insightface_root or os.environ.get("INSIGHTFACE_HOME") or os.path.expanduser("~/.insightface")
            providers = _onnx_providers()
            app = FaceAnalysis(
                name=settings.insightface_model,
                root=root,
                providers=providers,
                allowed_modules=["detection"],
            )
            app.prepare(
                ctx_id=settings.insightface_ctx_id,
                det_size=settings.insightface_det_size,
            )
            _face_app = app
        finally:
            os.dup2(_saved_fd, _stderr_fd)
            os.close(_saved_fd)
            _devnull.close()
        logger.info("InsightFace model loaded", extra={"model": get_settings().insightface_model})
    return _face_app


def decode_image(image_b64: str) -> np.ndarray | None:
    """Decode base64 image (with optional data URI) to OpenCV BGR."""
    if not image_b64 or not image_b64.strip():
        return None
    s = image_b64.strip()
    if s.startswith("data:"):
        s = s.split(",", 1)[1] if "," in s else s
    try:
        raw = base64.b64decode(s, validate=True)
    except Exception:
        return None
    npy = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(npy, cv2.IMREAD_COLOR)
    return img if img is not None and img.size > 0 else None


class LivenessResult:
    """Structured liveness result."""

    __slots__ = ("live", "confidence", "details", "errors")

    def __init__(
        self,
        *,
        live: bool = False,
        confidence: float = 0.0,
        details: dict[str, Any] | None = None,
        errors: list[str] | None = None,
    ):
        self.live = live
        self.confidence = confidence
        self.details = details or {}
        self.errors = errors or []

    def to_dict(self) -> dict[str, Any]:
        return {
            "live": self.live,
            "confidence": round(float(self.confidence), 4),
            "details": _to_native(self.details),
            "errors": list(self.errors),
        }


class LivenessService:
    """InsightFace-based liveness check service."""

    def __init__(self) -> None:
        self._settings = get_settings()

    def check(
        self,
        cv_img: np.ndarray,
        *,
        force_context_antispoof: bool = False,
        antispoof_real_threshold_override: float | None = None,
        antispoof_context_real_threshold_override: float | None = None,
        antispoof_min_logit_diff_override: float | None = None,
        liveness_confidence_threshold_override: float | None = None,
    ) -> LivenessResult:
        """
        Run liveness check on a single BGR image.
        Uses InsightFace for face detection and sharpness/size heuristics for liveness.

        force_context_antispoof: run anti-spoof on large scene crop (min with face crop score).
        Overrides apply to anti-spoof gating and/or live gate (e.g. motion sequence per-frame).
        """
        result = LivenessResult()
        if cv_img is None or cv_img.size == 0:
            result.errors.append("Invalid or empty image")
            return result

        try:
            real_thr = (
                antispoof_real_threshold_override
                if antispoof_real_threshold_override is not None
                else self._settings.antispoof_real_threshold
            )
            logit_min = (
                antispoof_min_logit_diff_override
                if antispoof_min_logit_diff_override is not None
                else self._settings.antispoof_min_logit_diff
            )
            ctx_thr = (
                antispoof_context_real_threshold_override
                if antispoof_context_real_threshold_override is not None
                else self._settings.antispoof_context_real_threshold
            )
            strategy = self._settings.antispoof_dual_crop_strategy

            # 1) Sharpness (anti-blur / anti-print)
            gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
            lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            result.details["laplacian_variance"] = round(lap_var, 2)
            sharpness = min(
                1.0,
                lap_var / self._settings.laplacian_min,
            ) if self._settings.laplacian_min > 0 else 1.0
            result.details["sharpness_score"] = round(sharpness, 4)

            # 2) Face detection via InsightFace (Direct)
            face_app = _get_face_app()
            faces = face_app.get(cv_img)
            min_det = self._settings.min_det_score
            faces = [f for f in faces if getattr(f, "det_score", 0) >= min_det]
            result.details["face_count"] = len(faces)
            result.details["detection_backend"] = "insightface"

            if not faces:
                result.details["reason"] = "No face detected"
                result.confidence = 0.0
                return result

            # Use highest-confidence face
            best = max(faces, key=lambda f: getattr(f, "det_score", 0))
            det_score = getattr(best, "det_score", 1.0)
            result.details["best_det_score"] = round(float(det_score), 4)
            bbox = getattr(best, "bbox", None)
            if bbox is not None:
                x1, y1, x2, y2 = map(int, bbox[:4])
                result.details["bbox"] = [x1, y1, x2 - x1, y2 - y1]

            h, w = cv_img.shape[:2]
            area_img = w * h
            if bbox is not None:
                x1, y1, x2, y2 = bbox[:4]
                face_area = (x2 - x1) * (y2 - y1)
                face_ratio = face_area / area_img
            else:
                face_ratio = 0.0
            result.details["largest_face_area_ratio"] = round(face_ratio, 4)

            size_ok = face_ratio >= self._settings.face_area_min_ratio
            sharp_ok = lap_var >= self._settings.laplacian_min

            # Heuristic component: sharpness + size + detection score
            size_component = min(1.0, face_ratio / 0.1) if size_ok else 0.0
            heuristic_conf = (
                0.4 * sharpness
                + 0.3 * size_component
                + 0.3 * float(det_score)
            )
            if not sharp_ok:
                heuristic_conf *= 0.7
            result.details["heuristic_confidence"] = round(heuristic_conf, 4)

            # Anti-spoof ONNX on face crop + optional large context (dual crop).
            antispoof_real = 0.0
            antispoof_used = False
            spoof_gate_ok = True
            pass_face = True
            pass_ctx = True
            face_real = 0.0
            ctx_real: float | None = None
            face_logit_ok = True
            use_ctx = False
            if self._settings.antispoof_enabled and bbox is not None:
                # Normalize bbox so x1<=x2, y1<=y2 (InsightFace can sometimes return inverted coords)
                x1, y1, x2, y2 = map(float, bbox[:4])
                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = min(y1, y2), max(y1, y2)
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                h_img, w_img = cv_img.shape[:2]
                face_w, face_h = max(1, x2 - x1), max(1, y2 - y1)
                pad_ratio = self._settings.antispoof_crop_padding_ratio
                pad = int(max(face_w, face_h) * pad_ratio)
                x1_c = max(0, x1 - pad)
                y1_c = max(0, y1 - pad)
                x2_c = min(w_img, x2 + pad)
                y2_c = min(h_img, y2 + pad)
                if x2_c > x1_c and y2_c > y1_c:
                    crop = cv_img[y1_c:y2_c, x1_c:x2_c]
                else:
                    crop = np.array([])
                if crop.size > 0:
                    face_real, antispoof_details = run_antispoof(crop)
                    result.details.update(antispoof_details)
                    antispoof_used = antispoof_details.get("antispoof") == "enabled"
                    face_ld = antispoof_details.get("antispoof_logit_diff")
                    if antispoof_used and logit_min > -999 and face_ld is not None:
                        face_logit_ok = bool(face_ld >= logit_min)

                    use_ctx = bool(self._settings.antispoof_dual_crop or force_context_antispoof)
                    if antispoof_used and use_ctx:
                        ctx_ratio = getattr(self._settings, "antispoof_context_padding_ratio", 2.0)
                        pad_ctx = int(max(face_w, face_h) * ctx_ratio)
                        x1_ctx = max(0, x1 - pad_ctx)
                        y1_ctx = max(0, y1 - pad_ctx)
                        x2_ctx = min(w_img, x2 + pad_ctx)
                        y2_ctx = min(h_img, y2 + pad_ctx)
                        if x2_ctx > x1_ctx and y2_ctx > y1_ctx:
                            crop_ctx = cv_img[y1_ctx:y2_ctx, x1_ctx:x2_ctx]
                            if crop_ctx.size > 0:
                                ctx_real, ctx_details = run_antispoof(crop_ctx)
                                result.details["antispoof_context_real_score"] = round(ctx_real, 4)
                                result.details["antispoof_context_logit_diff"] = ctx_details.get(
                                    "antispoof_logit_diff"
                                )
                                ctx_ld = ctx_details.get("antispoof_logit_diff")
                                ctx_logit_ok = True
                                if strategy == "min" and logit_min > -999 and ctx_ld is not None:
                                    ctx_logit_ok = bool(ctx_ld >= logit_min)

                                if strategy == "asymmetric":
                                    pass_face = bool(face_real >= real_thr and face_logit_ok)
                                    # Context: score gate only (logit is noisy on scene); min mode uses ctx_logit_ok below
                                    pass_ctx = bool(ctx_real >= ctx_thr)
                                    antispoof_real = 0.5 * (face_real + ctx_real)
                                    spoof_gate_ok = pass_face and pass_ctx
                                    result.details["antispoof_dual_crop_strategy"] = "asymmetric"
                                    result.details["antispoof_context_threshold_used"] = round(ctx_thr, 4)
                                    result.details["antispoof_face_gate_ok"] = pass_face
                                    result.details["antispoof_context_gate_ok"] = pass_ctx
                                else:
                                    # min: strict min of scores; logit must pass on both crops when enabled
                                    m = min(face_real, ctx_real)
                                    if not face_logit_ok or not ctx_logit_ok:
                                        m = 0.0
                                    antispoof_real = m
                                    spoof_gate_ok = m >= real_thr
                                    result.details["antispoof_dual_crop_strategy"] = "min"
                            else:
                                # Context region empty — fall back to face-only gating
                                antispoof_real = face_real if face_logit_ok else 0.0
                                spoof_gate_ok = bool(face_real >= real_thr and face_logit_ok)
                                result.details["antispoof_dual_crop_strategy"] = f"{strategy}_no_context_crop"
                        else:
                            antispoof_real = face_real if face_logit_ok else 0.0
                            spoof_gate_ok = bool(face_real >= real_thr and face_logit_ok)
                            result.details["antispoof_dual_crop_strategy"] = f"{strategy}_no_context_bounds"
                    else:
                        # Face crop only
                        antispoof_real = face_real if face_logit_ok else 0.0
                        spoof_gate_ok = antispoof_real >= real_thr
                        result.details["antispoof_dual_crop_strategy"] = "face_only"

            # Blend: antispoof_weight * antispoof_real + (1 - antispoof_weight) * heuristic
            w = self._settings.antispoof_weight if antispoof_used else 0.0
            confidence = (w * antispoof_real + (1.0 - w) * heuristic_conf) if w < 1.0 else antispoof_real
            if not antispoof_used:
                confidence = heuristic_conf
            result.confidence = min(1.0, confidence)

            if not sharp_ok:
                result.details["reason"] = "Image too blurry (possible photo/screen)"
            elif not size_ok:
                result.details["reason"] = "Face too small (possible screenshot)"
            elif antispoof_used and not spoof_gate_ok:
                if (
                    strategy == "asymmetric"
                    and use_ctx
                    and ctx_real is not None
                    and result.details.get("antispoof_dual_crop_strategy") == "asymmetric"
                ):
                    if not pass_face:
                        result.details["reason"] = (
                            "Anti-spoof: face logit margin too low"
                            if not face_logit_ok
                            else "Anti-spoof: face crop below threshold"
                        )
                    else:
                        result.details["reason"] = (
                            "Anti-spoof: context/scene suggests replay or screen (below context threshold)"
                        )
                else:
                    result.details["reason"] = "Anti-spoof: classified as spoof"
            else:
                result.details["reason"] = "OK"

            live_conf_thr = (
                liveness_confidence_threshold_override
                if liveness_confidence_threshold_override is not None
                else self._settings.liveness_confidence_threshold
            )
            result.live = (
                result.confidence >= live_conf_thr
                and sharp_ok
                and size_ok
                and (not antispoof_used or spoof_gate_ok)
            )
        except Exception as e:
            logger.exception("Liveness check failed")
            result.errors.append("Liveness check failed")
            # Do not expose traceback or internal errors in API response

        return result


def get_liveness_service() -> LivenessService:
    """Factory for dependency injection."""
    return LivenessService()
