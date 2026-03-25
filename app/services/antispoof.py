"""
Anti-spoof ONNX model (MiniFAS-style): 128x128 RGB input, real vs spoof score.
Compatible with SuriAI/face-antispoof-onnx and similar MiniFASNet ONNX exports.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.config import get_settings
from app.logging_config import get_logger

logger = get_logger(__name__)

_ort_session: Any = None


def _get_antispoof_session():
    """Lazy-load ONNX Runtime session for anti-spoof model (singleton)."""
    global _ort_session
    if _ort_session is not None:
        return _ort_session
    settings = get_settings()
    path = settings.antispoof_model_path
    if not path or not os.path.isfile(path):
        logger.warning("Anti-spoof model not found at %s; anti-spoof disabled", path)
        return None
    try:
        # Redirect stderr fd so ONNX Runtime C++ output is suppressed (cpuid warning on Mac/ARM/Docker)
        import sys
        _stderr_fd = getattr(sys.stderr, "fileno", lambda: 2)()
        _devnull = open(os.devnull, "w")
        _saved_fd = os.dup(_stderr_fd)
        try:
            os.dup2(_devnull.fileno(), _stderr_fd)
            import onnxruntime as ort
            if get_settings().use_gpu:
                available = set(ort.get_available_providers())
                providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in available] or ["CPUExecutionProvider"]
            else:
                providers = ["CPUExecutionProvider"]
            _ort_session = ort.InferenceSession(path, providers=providers)
        finally:
            os.dup2(_saved_fd, _stderr_fd)
            os.close(_saved_fd)
            _devnull.close()
        logger.info("Anti-spoof ONNX model loaded: %s", path)
        return _ort_session
    except Exception as e:
        logger.exception("Failed to load anti-spoof ONNX: %s", e)
        return None


def preprocess_face_crop(bgr_crop: np.ndarray, size: int = 128) -> np.ndarray:
    """
    Preprocess face crop to match SuriAI/face-antispoof-onnx: letterbox to size x size,
    normalize [0,1], NCHW. SuriAI uses INTER_LANCZOS4/INTER_AREA and BORDER_REFLECT_101.
    Output order from model: index 0 = REAL, index 1 = SPOOF (see SuriAI inference.py).
    """
    if bgr_crop is None or bgr_crop.size == 0:
        raise ValueError("Empty face crop")
    rgb = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)
    old_h, old_w = rgb.shape[:2]
    ratio = float(size) / max(old_h, old_w)
    new_w = int(old_w * ratio)
    new_h = int(old_h * ratio)
    interp = cv2.INTER_LANCZOS4 if ratio > 1.0 else cv2.INTER_AREA
    img = cv2.resize(rgb, (new_w, new_h), interpolation=interp)
    # Pad to size x size (letterbox) with reflection like SuriAI
    top = (size - new_h) // 2
    bottom = size - new_h - top
    left = (size - new_w) // 2
    right = size - new_w - left
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_REFLECT_101)
    x = img.astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))
    x = np.expand_dims(x, axis=0)
    return x


def run_antispoof(bgr_face_crop: np.ndarray) -> tuple[float, dict[str, Any]]:
    """
    Run anti-spoof model on a single face crop (BGR).
    Returns (real_score, details_dict). real_score in [0,1]; 1 = real, 0 = spoof.
    If model unavailable, returns (0.0, {"antispoof": "disabled"}).
    """
    details: dict[str, Any] = {}
    session = _get_antispoof_session()
    if session is None:
        details["antispoof"] = "disabled"
        return 0.0, details

    settings = get_settings()
    size = settings.antispoof_input_size
    try:
        input_tensor = preprocess_face_crop(bgr_face_crop, size=size)
        input_name = session.get_inputs()[0].name
        out = session.run(None, {input_name: input_tensor})
        # SuriAI model: index 0 = REAL, index 1 = SPOOF (see their inference.py process_with_logits)
        logits = out[0]  # (1, 2) -> [real_logit, spoof_logit]
        if logits.shape[-1] >= 2:
            real_logit = float(logits[0][0])
            spoof_logit = float(logits[0][1])
            logit_diff = real_logit - spoof_logit
            details["antispoof_real_logit"] = round(real_logit, 4)
            details["antispoof_spoof_logit"] = round(spoof_logit, 4)
            details["antispoof_logit_diff"] = round(logit_diff, 4)
            # Softmax for probability; SuriAI thresholds on logit_diff = real - spoof
            exp = np.exp(logits - logits.max(axis=-1, keepdims=True))
            probs = exp / exp.sum(axis=-1, keepdims=True)
            real_idx = get_settings().antispoof_real_index
            real_score = float(probs[0][real_idx])
            spoof_score = float(probs[0][1 - real_idx])
        else:
            real_score = float(logits[0][0])
            spoof_score = 1.0 - real_score
            logit_diff = 0.0
        details["antispoof_real_score"] = round(real_score, 4)
        details["antispoof_spoof_score"] = round(spoof_score, 4)
        details["antispoof"] = "enabled"
        return real_score, details
    except Exception as e:
        logger.warning("Anti-spoof inference failed: %s", e)
        details["antispoof"] = "error"
        details["antispoof_error"] = str(e)
        return 0.0, details
