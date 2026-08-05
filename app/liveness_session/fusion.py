"""Verdict fusion: pure function combining all gates into (live, reasons)."""

from __future__ import annotations


def decide_verdict(
    *,
    expired: bool,
    passive_pass: bool,
    challenge_done: bool,
    tampering_suspected: bool,
    spoof_suspected: bool,
    multiple_faces: bool,
    face_lost: bool,
) -> tuple[bool, list[str]]:
    """
    live = passive_pass AND challenge_done AND no anti-gaming failure AND not expired.
    Reasons use the fixed vocabulary:
    expired | challenge_failed | spoof_suspected | multiple_faces | face_lost | frame_tampering_suspected
    """
    reasons: list[str] = []
    if expired:
        reasons.append("expired")
    if tampering_suspected:
        reasons.append("frame_tampering_suspected")
    if multiple_faces:
        reasons.append("multiple_faces")
    if face_lost:
        reasons.append("face_lost")
    if spoof_suspected or (not passive_pass and not face_lost):
        # Passive gate failing (with a face present) is a spoof signal.
        reasons.append("spoof_suspected")
    if not challenge_done:
        reasons.append("challenge_failed")
    live = not reasons
    return live, reasons
