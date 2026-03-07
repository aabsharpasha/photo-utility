"""
Liveness check using InsightFace (Direct) + anti-spoof ONNX.

- Face detection: InsightFace buffalo_l (RetinaFace).
- Liveness: sharpness (Laplacian), face size, detection score + anti-spoof ONNX (MiniFAS-style).
"""

from __future__ import annotations

import base64
import os
import sys
import traceback
from typing import Any

import cv2
import numpy as np


def _to_native(x: Any) -> Any:
    """Convert numpy scalars/arrays in structures to native Python for JSON serialization."""
    if isinstance(x, (np.floating, np.integer)):
        return float(x) if isinstance(x, np.floating) else int(x)
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
    """Use CPU only unless USE_GPU=1, to avoid CUDA warning and OOM in Docker."""
    if os.environ.get("USE_GPU", "").strip().lower() in ("1", "true", "yes"):
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

    def check(self, cv_img: np.ndarray) -> LivenessResult:
        """
        Run liveness check on a single BGR image.
        Uses InsightFace for face detection and sharpness/size heuristics for liveness.
        """
        result = LivenessResult()
        if cv_img is None or cv_img.size == 0:
            result.errors.append("Invalid or empty image")
            return result

        try:
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

            # Anti-spoof ONNX on cropped face (larger padding = more context to detect photo/screen held to camera)
            antispoof_real = 0.0
            antispoof_used = False
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
                    antispoof_real, antispoof_details = run_antispoof(crop)
                    result.details.update(antispoof_details)
                    antispoof_used = antispoof_details.get("antispoof") == "enabled"
                    # Optional: require min logit_diff (SuriAI-style) for stricter spoof rejection
                    if antispoof_used and self._settings.antispoof_min_logit_diff > -999:
                        min_logit_diff = self._settings.antispoof_min_logit_diff
                        logit_diff = antispoof_details.get("antispoof_logit_diff", 0.0)
                        if logit_diff < min_logit_diff:
                            antispoof_real = 0.0  # Force fail so result.live is False

                    # Dual crop: also run on large-context crop (face + scene); use min score so photo/screen edges can trigger spoof
                    if antispoof_used and getattr(self._settings, "antispoof_dual_crop", False):
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
                                result.details["antispoof_context_logit_diff"] = ctx_details.get("antispoof_logit_diff")
                                antispoof_real = min(antispoof_real, ctx_real)
                                if ctx_details.get("antispoof_logit_diff") is not None and self._settings.antispoof_min_logit_diff > -999:
                                    if ctx_details["antispoof_logit_diff"] < self._settings.antispoof_min_logit_diff:
                                        antispoof_real = 0.0

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
            elif antispoof_used and antispoof_real < self._settings.antispoof_real_threshold:
                result.details["reason"] = "Anti-spoof: classified as spoof"
            else:
                result.details["reason"] = "OK"

            result.live = (
                result.confidence >= self._settings.liveness_confidence_threshold
                and sharp_ok
                and size_ok
                and (not antispoof_used or antispoof_real >= self._settings.antispoof_real_threshold)
            )
        except Exception as e:
            logger.exception("Liveness check failed")
            result.errors.append("Liveness check failed")
            # Do not expose traceback or internal errors in API response

        return result


def get_liveness_service() -> LivenessService:
    """Factory for dependency injection."""
    return LivenessService()
