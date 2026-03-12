"""
Face match / face verification service using InsightFace embeddings.

The API shape is intentionally aligned with AWS Rekognition CompareFaces:
- Request contains SourceImage, TargetImage, SimilarityThreshold.
- Response contains SourceImageFace, FaceMatches, UnmatchedFaces.
"""

from __future__ import annotations

import os
import sys
from typing import Any

import numpy as np

from app.config import get_settings
from app.logging_config import get_logger
from app.services.liveness import _onnx_providers, _to_native

logger = get_logger(__name__)
_face_match_app: Any = None


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1D vectors."""
    if a is None or b is None:
        return 0.0
    if a.ndim != 1 or b.ndim != 1:
        return 0.0
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1e-9
    return float(np.dot(a, b) / denom)


class FaceMatchService:
    """Face match / verification built on InsightFace embeddings."""

    def __init__(self) -> None:
        self._settings = get_settings()
        # Ensure recognition-capable model is lazy-loaded for face embeddings.
        self._face_app = _get_face_match_app()

    def compare(
        self,
        source_img: np.ndarray,
        target_img: np.ndarray,
        similarity_threshold: float | None = None,
    ) -> dict[str, Any]:
        """
        Compare faces between source and target images.

        Returns AWS Rekognition-style payload:
        {
          "SourceImageFace": { "BoundingBox": {...}, "Confidence": float },
          "FaceMatches": [ { "Similarity": float, "Face": {...} }, ... ],
          "UnmatchedFaces": [ { "BoundingBox": {...}, "Confidence": float }, ... ]
        }
        """
        if similarity_threshold is None:
            # Rekognition uses 0–100 for SimilarityThreshold; we mirror that scale.
            similarity_threshold = float(
                getattr(self._settings, "face_match_default_similarity_threshold", 70.0)
            )

        app = self._face_app

        # Detect faces and extract embeddings
        src_faces = app.get(source_img) if source_img is not None else []
        tgt_faces = app.get(target_img) if target_img is not None else []

        result: dict[str, Any] = {
            "SourceImageFace": None,
            "FaceMatches": [],
            "UnmatchedFaces": [],
        }

        if not src_faces:
            result["SourceImageFace"] = None
            result["FaceMatches"] = []
            result["UnmatchedFaces"] = [
                {
                    "BoundingBox": _bbox_to_relative(f.bbox, target_img.shape if target_img is not None else None),
                    "Confidence": float(getattr(f, "det_score", 0.0) * 100.0),
                }
                for f in tgt_faces
            ]
            return _to_native(result)

        # Choose the highest-confidence source face
        src_face = max(src_faces, key=lambda f: getattr(f, "det_score", 0.0))
        src_bbox = _bbox_to_relative(src_face.bbox, source_img.shape if source_img is not None else None)
        src_conf = float(getattr(src_face, "det_score", 0.0) * 100.0)
        result["SourceImageFace"] = {
            "BoundingBox": src_bbox,
            "Confidence": src_conf,
        }
        src_emb = getattr(src_face, "normed_embedding", None)
        src_emb = np.asarray(src_emb, dtype=np.float32) if src_emb is not None else None
        if src_emb is None or src_emb.size == 0:
            logger.warning("Source embedding unavailable; check recognition model initialization")
            return _to_native(result)

        # Build matches / unmatched for target faces
        for tf in tgt_faces:
            t_bbox = _bbox_to_relative(tf.bbox, target_img.shape if target_img is not None else None)
            t_conf = float(getattr(tf, "det_score", 0.0) * 100.0)
            t_emb = getattr(tf, "normed_embedding", None)
            t_emb = np.asarray(t_emb, dtype=np.float32) if t_emb is not None else None
            if t_emb is None or t_emb.size == 0:
                result["UnmatchedFaces"].append(
                    {
                        "BoundingBox": t_bbox,
                        "Confidence": t_conf,
                    }
                )
                continue

            similarity = _cosine_similarity(src_emb, t_emb) * 100.0  # nominally -100..100
            similarity = max(0.0, min(100.0, similarity))

            if similarity >= similarity_threshold:
                result["FaceMatches"].append(
                    {
                        "Similarity": float(similarity),
                        "Face": {
                            "BoundingBox": t_bbox,
                            "Confidence": t_conf,
                        },
                    }
                )
            else:
                result["UnmatchedFaces"].append(
                    {
                        "BoundingBox": t_bbox,
                        "Confidence": t_conf,
                    }
                )

        # Sort matches by similarity desc for deterministic output
        result["FaceMatches"].sort(key=lambda m: m.get("Similarity", 0.0), reverse=True)

        return _to_native(result)


def _bbox_to_relative(bbox: Any, img_shape: tuple[int, int, int] | None) -> dict[str, float] | None:
    """
    Convert InsightFace bbox [x1, y1, x2, y2] to Rekognition-style relative BoundingBox.
    """
    if bbox is None or img_shape is None:
        return None

    try:
        x1, y1, x2, y2 = map(float, bbox[:4])
    except Exception:
        return None

    height, width = img_shape[0], img_shape[1]
    if width <= 0 or height <= 0:
        return None

    # Normalize coordinates into [0, 1]
    left = max(0.0, min(1.0, x1 / width))
    top = max(0.0, min(1.0, y1 / height))
    w = max(0.0, min(1.0, (x2 - x1) / width))
    h = max(0.0, min(1.0, (y2 - y1) / height))

    return {
        "Left": left,
        "Top": top,
        "Width": w,
        "Height": h,
    }


def _get_face_match_app():
    """Lazy-load InsightFace FaceAnalysis with recognition enabled."""
    global _face_match_app
    if _face_match_app is None:
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
                # Face match needs embeddings, so recognition must be enabled.
                allowed_modules=["detection", "recognition"],
            )
            app.prepare(
                ctx_id=settings.insightface_ctx_id,
                det_size=settings.insightface_det_size,
            )
            _face_match_app = app
        finally:
            os.dup2(_saved_fd, _stderr_fd)
            os.close(_saved_fd)
            _devnull.close()
        logger.info("InsightFace face-match model loaded", extra={"model": get_settings().insightface_model})
    return _face_match_app


def get_face_match_service() -> FaceMatchService:
    """Factory for dependency injection."""
    return FaceMatchService()

