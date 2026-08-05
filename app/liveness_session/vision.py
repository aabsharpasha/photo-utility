"""
Model access for the session stream (blocking; call via run_in_executor).

Reuses the repo's existing models instead of adding new dependencies:
- InsightFace FaceAnalysis with the landmark_3d_68 module (buffalo_l pack) for
  detection + iBUG-68 landmarks + yaw (instead of MediaPipe).
- The existing MiniFAS ONNX anti-spoof (app/services/antispoof.run_antispoof)
  for passive PAD (instead of deepface/TensorFlow).
- The existing FFT moiré heuristic (app/services/replay_guard).
"""

from __future__ import annotations

import os
import sys
import threading
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from app.config import get_settings
from app.logging_config import get_logger
from app.services.antispoof import run_antispoof
from app.services.replay_guard import moire_screen_score_bgr

from app.liveness_session.active import ear_from_landmarks_68, estimate_yaw_degrees

logger = get_logger(__name__)

_face_app = None
_face_app_lock = threading.Lock()


def _get_landmark_face_app():
    """Lazy singleton FaceAnalysis with detection + 68-point 3D landmarks (adds pose/yaw)."""
    global _face_app
    if _face_app is None:
        with _face_app_lock:
            if _face_app is None:
                # Same stderr-suppression dance as app/services/liveness._get_face_app
                _stderr_fd = sys.stderr.fileno() if hasattr(sys.stderr, "fileno") else 2
                _devnull = open(os.devnull, "w")
                _saved_fd = os.dup(_stderr_fd)
                try:
                    os.dup2(_devnull.fileno(), _stderr_fd)
                    from insightface.app import FaceAnalysis

                    settings = get_settings()
                    root = (
                        settings.insightface_root
                        or os.environ.get("INSIGHTFACE_HOME")
                        or os.path.expanduser("~/.insightface")
                    )
                    from app.services.liveness import _onnx_providers

                    app = FaceAnalysis(
                        name=settings.insightface_model,
                        root=root,
                        providers=_onnx_providers(),
                        allowed_modules=["detection", "landmark_3d_68"],
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
                logger.info("Session liveness models loaded (detection + landmark_3d_68)")
    return _face_app


def warmup() -> None:
    """Load models once and run a dummy inference (called from /api/ready)."""
    app = _get_landmark_face_app()
    dummy = np.zeros((64, 64, 3), dtype=np.uint8)
    app.get(dummy)
    run_antispoof(dummy)


@dataclass
class FrameAnalysis:
    face_count: int
    bbox: Optional[tuple[float, float, float, float]] = None
    ear: Optional[float] = None
    yaw_degrees: Optional[float] = None
    antispoof_real: Optional[float] = None
    moire_score: Optional[float] = None


def analyze_frame(jpeg_bytes: bytes, *, run_passive: bool) -> Optional[FrameAnalysis]:
    """Decode + detect + landmarks (+ anti-spoof/moiré on passive frames). Returns None on undecodable input."""
    arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None or bgr.size == 0:
        return None

    settings = get_settings()
    faces = _get_landmark_face_app().get(bgr)
    faces = [f for f in faces if getattr(f, "det_score", 0) >= settings.min_det_score]
    if len(faces) != 1:
        return FrameAnalysis(face_count=len(faces))

    face = faces[0]
    x1, y1, x2, y2 = map(float, face.bbox[:4])
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)

    ear = None
    yaw = None
    lmk = getattr(face, "landmark_3d_68", None)
    if lmk is not None:
        pts = [(float(p[0]), float(p[1])) for p in lmk]
        ear = ear_from_landmarks_68(pts)
        pose = getattr(face, "pose", None)
        yaw = float(pose[1]) if pose is not None else estimate_yaw_degrees(pts)

    antispoof_real = None
    moire = None
    if run_passive:
        h, w = bgr.shape[:2]
        pad = int(max(x2 - x1, y2 - y1) * settings.antispoof_crop_padding_ratio)
        cx1, cy1 = max(0, int(x1) - pad), max(0, int(y1) - pad)
        cx2, cy2 = min(w, int(x2) + pad), min(h, int(y2) + pad)
        crop = bgr[cy1:cy2, cx1:cx2]
        if crop.size > 0:
            score, details = run_antispoof(crop)
            if details.get("antispoof") == "enabled":
                antispoof_real = float(score)
            moire = float(moire_screen_score_bgr(crop))

    return FrameAnalysis(
        face_count=1,
        bbox=(x1, y1, x2, y2),
        ear=ear,
        yaw_degrees=yaw,
        antispoof_real=antispoof_real,
        moire_score=moire,
    )
