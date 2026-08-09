"""Session-based liveness endpoints: create session, stream frames over WS, fetch verdict."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import os
import secrets
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, WebSocket, WebSocketDisconnect

from app.config import get_settings
from app.deps import require_api_key_query
from app.logging_config import get_logger, log_extra

from app.liveness_session.active import BlinkDetector, HeadTurnDetector
from app.liveness_session.antigaming import (
    DuplicateFrameDetector,
    FaceBoxTracker,
    MoireAggregator,
    SequenceValidator,
    is_jpeg,
    parse_framed_message,
)
from app.liveness_session.fusion import decide_verdict
from app.liveness_session.passive import PassiveAntispoofAggregator
from app.liveness_session.schemas import SessionCreateResponse, SessionResultResponse
from app.liveness_session.store import Session, get_session_store
from app.liveness_session import vision

logger = get_logger(__name__)
router = APIRouter()

CHALLENGES = ("blink", "turn_left", "turn_right")

_executor: ThreadPoolExecutor | None = None


def _get_executor() -> ThreadPoolExecutor:
    global _executor
    if _executor is None:
        workers = get_settings().liveness_session_inference_workers or (os.cpu_count() or 2)
        _executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="liveness-session")
    return _executor


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _analyze(jpeg_bytes: bytes, run_passive: bool):
    """Module-level indirection so tests can monkeypatch inference."""
    return vision.analyze_frame(jpeg_bytes, run_passive=run_passive)


@router.post(
    "/v1/liveness-sessions",
    response_model=SessionCreateResponse,
    tags=["liveness-session"],
    summary="Create a challenge–response liveness session",
    description=(
        "Creates a single-use session with a server-chosen random challenge (blink N times, turn left, "
        "or turn right) and returns a short-lived token for the frame-streaming WebSocket. "
        "The verdict is stored server-side; fetch it backend-to-backend via GET .../{id}/result."
    ),
)
async def create_session(
    _api_key: str = Depends(require_api_key_query),
) -> SessionCreateResponse:
    settings = get_settings()
    challenge = secrets.choice(CHALLENGES)
    # Fixed at 2: on slow cameras (~4 fps in dim rooms) each extra required blink is
    # another chance to miss the closed-eye frame; 3 blinks made those devices flaky.
    required_blinks = 2 if challenge == "blink" else 0
    session_id = uuid.uuid4().hex
    token = secrets.token_urlsafe(32)
    now = time.time()
    session = Session(
        id=session_id,
        token_sha256=_sha256(token),
        api_key_sha256=_sha256(_api_key),
        challenge=challenge,
        required_blinks=required_blinks,
        created_at=now,
        expires_at=now + settings.liveness_session_ttl_seconds,
    )
    get_session_store().put(session, settings.liveness_session_verdict_ttl_seconds)
    logger.info(
        "liveness session created",
        extra=log_extra(session_id=session_id, challenge=challenge),
    )
    return SessionCreateResponse(
        session_id=session_id,
        stream_token=token,
        challenge=challenge,
        required_blinks=required_blinks or None,
        expires_in_seconds=settings.liveness_session_ttl_seconds,
        stream_path=f"/api/v1/liveness-sessions/{session_id}/stream",
    )


@router.get(
    "/v1/liveness-sessions/{session_id}/result",
    response_model=SessionResultResponse,
    tags=["liveness-session"],
    summary="Fetch the stored session verdict (idempotent, backend-to-backend)",
)
async def session_result(
    session_id: str,
    _api_key: str = Depends(require_api_key_query),
) -> SessionResultResponse:
    session = get_session_store().get(session_id)
    if session is None or not hmac.compare_digest(_sha256(_api_key), session.api_key_sha256):
        raise HTTPException(status_code=404, detail="Unknown or expired session")
    status = session.status
    if status != "done" and session.expired():
        status = "expired"
    return SessionResultResponse(
        session_id=session.id,
        status=status,
        live=session.live,
        reasons=session.reasons,
    )


class _StreamState:
    """Mutable per-connection state; single consumer, no locking needed."""

    def __init__(self, session: Session) -> None:
        settings = get_settings()
        self.settings = settings
        self.session = session
        self.seq = SequenceValidator()
        self.dupes = DuplicateFrameDetector(
            max_identical=settings.liveness_session_max_identical_consecutive_frames
        )
        self.face_tracker = FaceBoxTracker(min_iou=settings.liveness_session_min_face_iou)
        self.moire = MoireAggregator(
            max_score=settings.liveness_session_moire_max_score,
            max_frame_fraction=settings.liveness_session_moire_max_frame_fraction,
        )
        self.passive = PassiveAntispoofAggregator(
            mean_threshold=settings.liveness_session_passive_mean_threshold,
            min_frames=settings.liveness_session_passive_min_frames,
        )
        self.blink: Optional[BlinkDetector] = None
        self.turn: Optional[HeadTurnDetector] = None
        if session.challenge == "blink":
            self.blink = BlinkDetector(
                required_blinks=session.required_blinks,
                ear_closed_threshold=settings.liveness_session_ear_closed_threshold,
                max_blinks_per_second=settings.liveness_session_max_blinks_per_second,
                ear_relative_drop=settings.liveness_session_ear_relative_drop,
            )
        else:
            self.turn = HeadTurnDetector(
                direction=session.challenge,
                turn_degrees=settings.liveness_session_yaw_turn_degrees,
                frontal_degrees=settings.liveness_session_yaw_frontal_degrees,
                min_consecutive_frames=settings.liveness_session_turn_min_consecutive_frames,
            )
        self.frames_received = 0
        self.frames_analyzed = 0
        self.consecutive_no_face = 0
        self.face_ever_seen = False
        self.multiple_faces = False
        self.face_lost = False
        self.tampering = False
        self.hint = "waiting for frames"

    @property
    def challenge_done(self) -> bool:
        if self.blink is not None:
            return self.blink.done and not self.blink.noise_detected
        return self.turn is not None and self.turn.done

    @property
    def spoof_suspected(self) -> bool:
        return self.moire.spoof_suspected or self.face_tracker.swap_suspected

    def verdict(self, *, expired: bool) -> tuple[bool, list[str]]:
        return decide_verdict(
            expired=expired,
            passive_pass=self.passive.passed,
            challenge_done=self.challenge_done,
            tampering_suspected=self.tampering
            or self.dupes.tampering_suspected
            or self.seq.violated,
            spoof_suspected=self.spoof_suspected,
            multiple_faces=self.multiple_faces,
            face_lost=self.face_lost or not self.face_ever_seen,
        )

    def progress_payload(self) -> dict:
        return {
            "type": "progress",
            "hint": self.hint,
            "challenge": self.session.challenge,
            "challenge_done": self.challenge_done,
            "blinks_detected": self.blink.blinks if self.blink else None,
            "blink_debug": self.blink.debug_state() if self.blink else None,
            "turn_state": self.turn.state if self.turn else None,
            "frames_received": self.frames_received,
            "passive_frames_analyzed": self.passive.frames_analyzed,
            "passive_pass": self.passive.passed,
        }


def _apply_analysis(state: _StreamState, analysis, timestamp: float) -> Optional[str]:
    """Update state machines with one frame's analysis; return a terminal failure reason or None."""
    settings = state.settings
    if analysis is None:
        state.hint = "frame could not be decoded"
        return None
    if analysis.face_count == 0:
        state.consecutive_no_face += 1
        state.face_tracker.reset()
        state.hint = "face not detected"
        if state.consecutive_no_face > settings.liveness_session_max_no_face_frames:
            state.face_lost = True
            return "face_lost"
        return None
    if analysis.face_count > 1:
        state.multiple_faces = True
        state.hint = "multiple faces"
        return "multiple_faces"

    state.consecutive_no_face = 0
    state.face_ever_seen = True
    state.hint = "ok"
    state.face_tracker.update(analysis.bbox)
    if state.face_tracker.swap_suspected:
        state.hint = "face changed abruptly"
        return "spoof_suspected"

    if analysis.antispoof_real is not None:
        state.passive.add(analysis.antispoof_real)
    if analysis.moire_score is not None:
        state.moire.add(analysis.moire_score)
        if state.moire.spoof_suspected:
            state.hint = "screen-like pattern detected"
            return "spoof_suspected"

    if state.blink is not None:
        if analysis.ear is not None:
            state.blink.update(analysis.ear, timestamp)
            d = state.blink.debug_state()
            logger.debug(
                "blink frame session=%s ear=%s thr=%s closed=%s blinks=%d noise=%s",
                state.session.id[:8],
                d["ear"],
                d["ear_threshold"],
                d["eyes_closed_now"],
                state.blink.blinks,
                d["noise_detected"],
            )
        else:
            # Face found but no usable landmarks: blink cannot progress on this frame.
            logger.debug(
                "blink frame session=%s ear=None (no landmarks)", state.session.id[:8]
            )
    if state.turn is not None and analysis.yaw_degrees is not None:
        state.turn.update(analysis.yaw_degrees)
    return None


@router.websocket("/v1/liveness-sessions/{session_id}/stream")
async def stream_session(
    ws: WebSocket,
    session_id: str,
    token: str = Query(default=""),
) -> None:
    """
    Client streams binary JPEG frames (5–10 fps), each prefixed with a 4-byte
    big-endian sequence number. Server runs passive + active + anti-gaming checks
    and pushes JSON progress messages, ending with a result message.
    """
    settings = get_settings()
    store = get_session_store()
    session = store.get(session_id)
    if (
        session is None
        or session.status != "pending"  # single-use
        or session.expired()
        or not token
        or not hmac.compare_digest(_sha256(token), session.token_sha256)
    ):
        await ws.close(code=1008)
        return

    await ws.accept()
    session.status = "streaming"
    store.put(session, settings.liveness_session_verdict_ttl_seconds)

    state = _StreamState(session)
    loop = asyncio.get_running_loop()
    inflight = 0
    finished_early = False
    failure_hint: Optional[str] = None

    try:
        while True:
            remaining = session.expires_at - time.time()
            if remaining <= 0:
                break
            try:
                message = await asyncio.wait_for(ws.receive_bytes(), timeout=remaining)
            except asyncio.TimeoutError:
                break

            state.frames_received += 1
            try:
                seq, jpeg = parse_framed_message(message)
            except ValueError:
                state.tampering = True
                failure_hint = "malformed frame message"
                break
            if not state.seq.check(seq):
                # Replay defense: out-of-order/duplicate sequence terminates the session.
                failure_hint = "sequence number replay detected"
                break
            if len(jpeg) > settings.liveness_session_max_frame_bytes or not is_jpeg(jpeg):
                state.tampering = True
                failure_hint = "invalid or oversized frame"
                break
            state.dupes.update(jpeg)
            if state.dupes.tampering_suspected:
                failure_hint = "static repeated frames"
                break

            # Backpressure: drop frames while the executor queue for this session is full.
            if inflight >= settings.liveness_session_max_inflight_frames:
                continue

            run_passive = state.frames_received % settings.liveness_session_passive_every_n == 0
            inflight += 1
            try:
                analysis = await loop.run_in_executor(
                    _get_executor(), _analyze, jpeg, run_passive
                )
            finally:
                inflight -= 1
            state.frames_analyzed += 1

            terminal = _apply_analysis(state, analysis, time.time())
            await ws.send_json(state.progress_payload())
            if terminal is not None:
                failure_hint = state.hint
                break
            if state.challenge_done and state.passive.passed:
                finished_early = True
                break
    except WebSocketDisconnect:
        pass
    except Exception:
        logger.exception("liveness session stream failed")

    expired = session.expired() and not finished_early
    live, reasons = state.verdict(expired=expired)
    session.status = "done"
    session.live = live
    session.reasons = reasons
    session.verdict_at = time.time()
    store.put(session, settings.liveness_session_verdict_ttl_seconds)
    logger.info(
        "liveness session verdict live=%s reasons=[%s] frames=%d analyzed=%d "
        "passive_analyzed=%d passive_mean=%.4f challenge_done=%s age=%.1fs session=%s",
        live,
        ",".join(reasons),
        state.frames_received,
        state.frames_analyzed,
        state.passive.frames_analyzed,
        state.passive.mean_score,
        state.challenge_done,
        time.time() - session.created_at,
        session.id,
    )
    if state.blink is not None:
        logger.info(
            "blink summary session=%s blinks=%d/%d %s",
            session.id,
            state.blink.blinks,
            state.blink.required_blinks,
            state.blink.debug_state(),
        )
    try:
        await ws.send_json(
            {
                "type": "result",
                "live": live,
                "reasons": reasons,
                "hint": failure_hint,
            }
        )
        await ws.close()
    except Exception:
        pass  # client already gone; verdict is stored and served via GET
