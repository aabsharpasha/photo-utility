"""Active challenge state machines: blink (EAR) and head turn (yaw). Pure classes."""

from __future__ import annotations

import math
from collections import deque
from typing import Sequence


def eye_aspect_ratio(eye_xy: Sequence[Sequence[float]]) -> float:
    """
    EAR for one eye given 6 (x, y) points in iBUG order:
    p1 outer corner, p2/p3 upper lid, p4 inner corner, p5/p6 lower lid.
    EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|).
    """
    if len(eye_xy) != 6:
        raise ValueError("EAR needs exactly 6 eye points")

    def d(a: Sequence[float], b: Sequence[float]) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    horiz = d(eye_xy[0], eye_xy[3])
    if horiz <= 1e-9:
        return 0.0
    return (d(eye_xy[1], eye_xy[5]) + d(eye_xy[2], eye_xy[4])) / (2.0 * horiz)


def ear_from_landmarks_68(landmarks_xy: Sequence[Sequence[float]]) -> float:
    """Mean EAR of both eyes from iBUG-68 landmarks (right eye 36-41, left eye 42-47)."""
    right = [landmarks_xy[i] for i in range(36, 42)]
    left = [landmarks_xy[i] for i in range(42, 48)]
    return (eye_aspect_ratio(right) + eye_aspect_ratio(left)) / 2.0


def estimate_yaw_degrees(landmarks_xy: Sequence[Sequence[float]]) -> float:
    """
    Geometric yaw fallback from iBUG-68 landmarks: nose tip (30) vs jaw edges (0, 16).
    Negative = subject turned left, positive = turned right (image coordinates).
    """
    nose = landmarks_xy[30]
    left_edge = landmarks_xy[0]
    right_edge = landmarks_xy[16]
    d_left = math.hypot(nose[0] - left_edge[0], nose[1] - left_edge[1])
    d_right = math.hypot(nose[0] - right_edge[0], nose[1] - right_edge[1])
    total = d_left + d_right
    if total <= 1e-9:
        return 0.0
    # r in [-1, 1]: 0 = frontal; clamp then map to degrees.
    r = max(-1.0, min(1.0, (d_left - d_right) / total))
    return math.degrees(math.asin(r))


class BlinkDetector:
    """
    Counts blinks as closed->open EAR transitions; transitions above
    max_blinks_per_second are treated as sensor/replay noise and not counted.

    Closure is judged RELATIVE to the person's rolling baseline EAR once enough
    samples exist (catches partial-closure frames at low fps and adapts to narrow
    eyes / camera angle); the absolute threshold is only the cold-start fallback.
    """

    _BASELINE_MIN_SAMPLES = 5

    def __init__(
        self,
        *,
        required_blinks: int,
        ear_closed_threshold: float,
        max_blinks_per_second: float,
        ear_relative_drop: float = 0.25,
    ) -> None:
        self.required_blinks = required_blinks
        self._closed_thr = ear_closed_threshold
        self._relative_drop = ear_relative_drop
        self._max_rate = max_blinks_per_second
        self._ears: deque[float] = deque(maxlen=24)
        self._baseline_was_ready = False
        self._was_closed = False
        self._blink_times: deque[float] = deque(maxlen=64)
        self.blinks = 0
        self.noise_detected = False
        # Debug/diagnostics (populated as frames arrive)
        self.last_ear: float | None = None
        self.last_threshold: float | None = None
        self.ear_min: float | None = None
        self.ear_max: float | None = None
        self.closed_frames = 0
        self.frames_seen = 0

    def _closed_threshold(self) -> float:
        if len(self._ears) < self._BASELINE_MIN_SAMPLES:
            return self._closed_thr
        baseline = sorted(self._ears)[len(self._ears) // 2]  # median: robust to blink dips
        return baseline * (1.0 - self._relative_drop)

    def update(self, ear: float, timestamp: float) -> None:
        ready = len(self._ears) >= self._BASELINE_MIN_SAMPLES
        threshold = self._closed_threshold()
        closed = ear < threshold
        self._ears.append(ear)
        self.frames_seen += 1
        self.last_ear = ear
        self.last_threshold = threshold
        self.ear_min = ear if self.ear_min is None else min(self.ear_min, ear)
        self.ear_max = ear if self.ear_max is None else max(self.ear_max, ear)
        if closed:
            self.closed_frames += 1
        if ready and not self._baseline_was_ready:
            # Absolute -> relative threshold switchover: resync state so users whose
            # resting EAR sits below the absolute fallback don't get a phantom blink.
            self._baseline_was_ready = True
            self._was_closed = closed
            return
        if self._was_closed and not closed:
            recent = [t for t in self._blink_times if timestamp - t <= 1.0]
            if len(recent) + 1 > self._max_rate:
                self.noise_detected = True
            else:
                self.blinks += 1
            self._blink_times.append(timestamp)
        self._was_closed = closed

    @property
    def done(self) -> bool:
        return self.blinks >= self.required_blinks

    def debug_state(self) -> dict:
        """Compact diagnostics for progress payloads and logs."""

        def r(v: float | None) -> float | None:
            return round(v, 4) if v is not None else None

        return {
            "ear": r(self.last_ear),
            "ear_threshold": r(self.last_threshold),
            "ear_min": r(self.ear_min),
            "ear_max": r(self.ear_max),
            "eyes_closed_now": self._was_closed,
            "closed_frames": self.closed_frames,
            "frames_seen": self.frames_seen,
            "noise_detected": self.noise_detected,
        }


class HeadTurnDetector:
    """
    turn_left / turn_right challenge: |yaw| > turn_degrees in the required direction
    for >= min_consecutive_frames, then a mandatory return to frontal
    (|yaw| < frontal_degrees) — the return defeats static side-profile photos.
    """

    def __init__(
        self,
        *,
        direction: str,  # "turn_left" | "turn_right"
        turn_degrees: float,
        frontal_degrees: float,
        min_consecutive_frames: int,
    ) -> None:
        if direction not in ("turn_left", "turn_right"):
            raise ValueError(f"Unknown turn direction: {direction}")
        self.direction = direction
        self._turn_deg = turn_degrees
        self._frontal_deg = frontal_degrees
        self._min_frames = min_consecutive_frames
        self._consecutive_turned = 0
        self.turned = False  # held the turn long enough
        self.done = False  # turned AND returned to frontal

    def update(self, yaw_degrees: float) -> None:
        if self.done:
            return
        if self.turned:
            if abs(yaw_degrees) < self._frontal_deg:
                self.done = True
            return
        in_direction = (
            yaw_degrees <= -self._turn_deg
            if self.direction == "turn_left"
            else yaw_degrees >= self._turn_deg
        )
        if in_direction:
            self._consecutive_turned += 1
            if self._consecutive_turned >= self._min_frames:
                self.turned = True
        else:
            self._consecutive_turned = 0

    @property
    def state(self) -> str:
        if self.done:
            return "done"
        if self.turned:
            return "awaiting_frontal"
        return "awaiting_turn"
