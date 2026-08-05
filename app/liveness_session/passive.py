"""Passive anti-spoof aggregation over streamed frames (pure logic, model-agnostic)."""

from __future__ import annotations

from collections import deque


class PassiveAntispoofAggregator:
    """Rolling window of anti-spoof real-scores; pass = mean >= threshold AND enough frames."""

    def __init__(self, *, mean_threshold: float, min_frames: int, window: int = 60) -> None:
        self._scores: deque[float] = deque(maxlen=window)
        self._mean_threshold = mean_threshold
        self._min_frames = min_frames

    def add(self, real_score: float) -> None:
        self._scores.append(float(real_score))

    @property
    def frames_analyzed(self) -> int:
        return len(self._scores)

    @property
    def mean_score(self) -> float:
        return sum(self._scores) / len(self._scores) if self._scores else 0.0

    @property
    def passed(self) -> bool:
        return (
            len(self._scores) >= self._min_frames
            and self.mean_score >= self._mean_threshold
        )
