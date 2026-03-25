"""
Lightweight signals for screen / video-replay presentation attacks.

These are heuristics (not a replacement for challenge–response or dedicated PAD models).
Use together with large-context anti-spoof on the motion endpoint.
"""

from __future__ import annotations

import cv2
import numpy as np


def moire_screen_score_bgr(bgr: np.ndarray) -> float:
    """
    Heuristic [0, 1] for display-style moiré / narrowband FFT peaks.

    p95/median on raw magnitude falsely spikes on almost any sharp natural photo or JPEG,
    so we use log-magnitude and *tail shape* (p99 vs p90 relative to IQR). Real webcams
    usually stay well below ~0.35; enable the gate only if you tune MOTION_MOIRE_MAX_SCORE.
    """
    if bgr is None or bgr.size == 0:
        return 0.0
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(gray, (256, 256), interpolation=cv2.INTER_AREA)
    g = cv2.GaussianBlur(g, (3, 3), 0)
    dft = cv2.dft(np.float32(g), flags=cv2.DFT_COMPLEX_OUTPUT)
    mag = cv2.magnitude(dft[:, :, 0], dft[:, :, 1])
    mag = np.fft.fftshift(mag)
    h, w = mag.shape
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    rad = np.hypot(x - cx, y - cy)
    ring = (rad >= 12) & (rad <= min(h, w) // 2 - 8)
    if not np.any(ring):
        return 0.0
    vals = mag[ring]
    # Compress dynamic range so a few hot bins don't dominate percentiles like p95/median did.
    lv = np.log(np.maximum(vals.astype(np.float64), 1e-10))
    p25 = float(np.percentile(lv, 25))
    p75 = float(np.percentile(lv, 75))
    p95 = float(np.percentile(lv, 95))
    p99 = float(np.percentile(lv, 99))
    p50 = float(np.percentile(lv, 50))
    iqr = max(p75 - p25, 0.04)
    # How much the top 1% pulls away from the 90–95% band vs typical spread (narrowband peaks).
    tail_spike = (p99 - p95) / iqr
    # Secondary: gentler broadband tail (log domain)
    broad = max(0.0, (p95 - p50) / iqr)
    s1 = float(np.tanh(max(0.0, tail_spike - 1.1) / 2.8))
    s2 = float(np.tanh(max(0.0, broad - 2.2) / 4.5))
    score = 0.78 * s1 + 0.22 * s2
    return float(np.clip(score, 0.0, 1.0))


def motion_sequence_replay_metrics(frames_bgr: list[np.ndarray]) -> dict:
    """Aggregate replay-oriented metrics over a frame sequence (diagnostics + optional gating)."""
    if not frames_bgr:
        return {"moire_scores": [], "moire_max": 0.0}
    scores = [round(moire_screen_score_bgr(im), 4) for im in frames_bgr]
    return {
        "moire_scores": scores,
        "moire_max": float(max(scores)) if scores else 0.0,
    }
