"""Anti-gaming checks for the frame stream: replay, tampering, face swap. Pure classes."""

from __future__ import annotations

import hashlib
from typing import Optional, Sequence

JPEG_MAGIC = b"\xff\xd8\xff"


def is_jpeg(data: bytes) -> bool:
    return data[:3] == JPEG_MAGIC


def parse_framed_message(data: bytes) -> tuple[int, bytes]:
    """Split a binary WS message into (sequence_number, jpeg_bytes); 4-byte big-endian prefix."""
    if len(data) < 5:
        raise ValueError("frame message too short")
    return int.from_bytes(data[:4], "big"), data[4:]


class SequenceValidator:
    """Sequence numbers must be strictly increasing; anything else is a replay signal."""

    def __init__(self) -> None:
        self._last: Optional[int] = None
        self.violated = False

    def check(self, seq: int) -> bool:
        if self._last is not None and seq <= self._last:
            self.violated = True
            return False
        self._last = seq
        return True


def bbox_iou(a: Sequence[float], b: Sequence[float]) -> float:
    """IoU of two [x1, y1, x2, y2] boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class FaceBoxTracker:
    """Consecutive face-box IoU must stay above min_iou (defeats mid-session photo swap)."""

    def __init__(self, *, min_iou: float) -> None:
        self._min_iou = min_iou
        self._prev: Optional[Sequence[float]] = None
        self.swap_suspected = False

    def update(self, bbox: Sequence[float]) -> None:
        if self._prev is not None and bbox_iou(self._prev, bbox) <= self._min_iou:
            self.swap_suspected = True
        self._prev = bbox

    def reset(self) -> None:
        """Call when the face is lost so re-acquisition isn't flagged as a swap."""
        self._prev = None


class DuplicateFrameDetector:
    """More than max_identical pixel-identical consecutive frames -> tampering suspected."""

    def __init__(self, *, max_identical: int) -> None:
        self._max = max_identical
        self._last_hash: Optional[str] = None
        self._run = 1
        self.tampering_suspected = False

    def update(self, jpeg_bytes: bytes) -> None:
        h = hashlib.sha256(jpeg_bytes).hexdigest()
        if h == self._last_hash:
            self._run += 1
            if self._run > self._max:
                self.tampering_suspected = True
        else:
            self._run = 1
        self._last_hash = h


class MoireAggregator:
    """Fraction of analyzed frames with a screen-like moiré score above threshold."""

    def __init__(self, *, max_score: float, max_frame_fraction: float) -> None:
        self._max_score = max_score
        self._max_fraction = max_frame_fraction
        self._total = 0
        self._flagged = 0

    def add(self, moire_score: float) -> None:
        self._total += 1
        if moire_score > self._max_score:
            self._flagged += 1

    @property
    def spoof_suspected(self) -> bool:
        return self._total > 0 and (self._flagged / self._total) > self._max_fraction
