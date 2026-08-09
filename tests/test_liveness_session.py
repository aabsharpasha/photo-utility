"""Unit tests for the session-based liveness feature (pure logic; models are not loaded)."""

import pytest

from app.liveness_session.active import (
    BlinkDetector,
    HeadTurnDetector,
    ear_from_landmarks_68,
    estimate_yaw_degrees,
    eye_aspect_ratio,
)
from app.liveness_session.antigaming import (
    DuplicateFrameDetector,
    FaceBoxTracker,
    MoireAggregator,
    SequenceValidator,
    bbox_iou,
    is_jpeg,
    parse_framed_message,
)
from app.liveness_session.fusion import decide_verdict
from app.liveness_session.passive import PassiveAntispoofAggregator


# ---------- EAR / yaw math ----------


def _eye(open_ratio: float):
    """Synthetic 6-point eye: width 10, lid distance controlled by open_ratio."""
    return [
        (0.0, 0.0),
        (3.0, -open_ratio),
        (7.0, -open_ratio),
        (10.0, 0.0),
        (7.0, open_ratio),
        (3.0, open_ratio),
    ]


def test_ear_open_vs_closed():
    open_ear = eye_aspect_ratio(_eye(2.0))  # (4+4)/(2*10) = 0.4
    closed_ear = eye_aspect_ratio(_eye(0.5))  # (1+1)/(2*10) = 0.1
    assert open_ear == pytest.approx(0.4)
    assert closed_ear == pytest.approx(0.1)


def test_ear_from_landmarks_68_uses_both_eyes():
    lmk = [(0.0, 0.0)] * 68
    for offset, base_x in ((36, 0.0), (42, 20.0)):
        for i, (x, y) in enumerate(_eye(2.0)):
            lmk[offset + i] = (base_x + x, y)
    assert ear_from_landmarks_68(lmk) == pytest.approx(0.4)


def test_estimate_yaw_frontal_and_turned():
    lmk = [(0.0, 0.0)] * 68
    lmk[0], lmk[16] = (0.0, 0.0), (10.0, 0.0)
    lmk[30] = (5.0, 2.0)  # centered nose -> ~0 deg
    assert abs(estimate_yaw_degrees(lmk)) < 1e-6
    lmk[30] = (9.0, 0.0)  # nose near right edge -> strongly positive
    assert estimate_yaw_degrees(lmk) > 20
    lmk[30] = (1.0, 0.0)
    assert estimate_yaw_degrees(lmk) < -20


# ---------- Blink state machine ----------


def test_blink_counting_and_completion():
    d = BlinkDetector(required_blinks=2, ear_closed_threshold=0.2, max_blinks_per_second=8.0)
    t = 0.0
    for ear in (0.3, 0.1, 0.3, 0.1, 0.3):  # two closed->open transitions
        d.update(ear, t)
        t += 0.5
    assert d.blinks == 2
    assert d.done
    assert not d.noise_detected


def test_blink_requires_reopen():
    d = BlinkDetector(required_blinks=1, ear_closed_threshold=0.2, max_blinks_per_second=8.0)
    for i, ear in enumerate((0.3, 0.1, 0.1, 0.1)):  # closes but never reopens
        d.update(ear, i * 0.1)
    assert d.blinks == 0
    assert not d.done


def test_blink_relative_drop_catches_partial_closure():
    """Baseline EAR 0.30 with dips only to 0.24: never crosses the absolute 0.20
    threshold, but is a >25% drop below the rolling median -> must count."""
    d = BlinkDetector(
        required_blinks=1, ear_closed_threshold=0.2, max_blinks_per_second=8.0, ear_relative_drop=0.25
    )
    t = 0.0
    for ear in (0.33, 0.33, 0.33, 0.33, 0.33, 0.24, 0.33):
        d.update(ear, t)
        t += 0.2
    assert d.blinks == 1
    assert d.done


def test_blink_narrow_eyes_low_baseline_not_stuck_closed():
    """Resting EAR 0.18 sits below the absolute threshold; the relative baseline
    must adapt so normal open eyes are not treated as permanently closed."""
    d = BlinkDetector(
        required_blinks=1, ear_closed_threshold=0.2, max_blinks_per_second=8.0, ear_relative_drop=0.25
    )
    t = 0.0
    for ear in (0.18, 0.18, 0.18, 0.18, 0.18, 0.18, 0.10, 0.18):
        d.update(ear, t)
        t += 0.2
    assert d.blinks == 1


def test_blink_noise_rejection_over_rate_limit():
    d = BlinkDetector(required_blinks=3, ear_closed_threshold=0.2, max_blinks_per_second=8.0)
    t = 0.0
    for _ in range(20):  # 50 Hz flicker: implausible blink rate
        d.update(0.1, t)
        t += 0.01
        d.update(0.3, t)
        t += 0.01
    assert d.noise_detected


# ---------- Head-turn state machine ----------


def test_turn_requires_hold_and_return_to_frontal():
    d = HeadTurnDetector(
        direction="turn_left", turn_degrees=20, frontal_degrees=10, min_consecutive_frames=3
    )
    for yaw in (-25, -25, -25):
        d.update(yaw)
    assert d.turned and not d.done  # held the turn, still profile
    d.update(-15)  # not frontal yet
    assert not d.done
    d.update(-5)
    assert d.done


def test_turn_wrong_direction_never_completes():
    d = HeadTurnDetector(
        direction="turn_left", turn_degrees=20, frontal_degrees=10, min_consecutive_frames=3
    )
    for yaw in (25, 25, 25, 5):
        d.update(yaw)
    assert not d.turned and not d.done


def test_turn_hold_resets_on_interruption():
    d = HeadTurnDetector(
        direction="turn_right", turn_degrees=20, frontal_degrees=10, min_consecutive_frames=3
    )
    for yaw in (25, 25, 0, 25, 25):  # never 3 consecutive
        d.update(yaw)
    assert not d.turned


def test_turn_static_profile_photo_fails():
    d = HeadTurnDetector(
        direction="turn_right", turn_degrees=20, frontal_degrees=10, min_consecutive_frames=3
    )
    for _ in range(50):  # side-profile photo held forever: turned, never frontal again
        d.update(30)
    assert d.turned and not d.done


# ---------- Anti-gaming ----------


def test_sequence_validator_rejects_replay_and_duplicates():
    v = SequenceValidator()
    assert v.check(1) and v.check(2) and v.check(10)
    assert not v.check(10)  # duplicate
    assert v.violated
    v2 = SequenceValidator()
    assert v2.check(5)
    assert not v2.check(3)  # out of order
    assert v2.violated


def test_parse_framed_message_and_jpeg_magic():
    payload = (7).to_bytes(4, "big") + b"\xff\xd8\xff" + b"rest"
    seq, jpeg = parse_framed_message(payload)
    assert seq == 7
    assert is_jpeg(jpeg)
    assert not is_jpeg(b"\x89PNG")
    with pytest.raises(ValueError):
        parse_framed_message(b"\x00\x00")


def test_bbox_iou_and_swap_detection():
    assert bbox_iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)
    assert bbox_iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    t = FaceBoxTracker(min_iou=0.3)
    t.update((0, 0, 10, 10))
    t.update((1, 1, 11, 11))  # small drift: fine
    assert not t.swap_suspected
    t.update((50, 50, 60, 60))  # photo swap: box jumps
    assert t.swap_suspected


def test_face_tracker_reset_after_face_lost():
    t = FaceBoxTracker(min_iou=0.3)
    t.update((0, 0, 10, 10))
    t.reset()
    t.update((50, 50, 60, 60))  # re-acquisition after loss is not a swap
    assert not t.swap_suspected


def test_duplicate_frame_detector():
    d = DuplicateFrameDetector(max_identical=3)
    for _ in range(3):
        d.update(b"same-frame")
    assert not d.tampering_suspected
    d.update(b"same-frame")  # 4th identical
    assert d.tampering_suspected


def test_moire_aggregator_fraction_gate():
    m = MoireAggregator(max_score=0.6, max_frame_fraction=0.3)
    for s in (0.1, 0.2, 0.1, 0.9):  # 25% flagged
        m.add(s)
    assert not m.spoof_suspected
    m.add(0.95)  # 40% flagged
    assert m.spoof_suspected


# ---------- Passive aggregator ----------


def test_passive_needs_min_frames_and_mean():
    p = PassiveAntispoofAggregator(mean_threshold=0.8, min_frames=8)
    for _ in range(7):
        p.add(0.95)
    assert not p.passed  # not enough frames
    p.add(0.95)
    assert p.passed
    for _ in range(20):
        p.add(0.1)  # spoofy frames drag the mean down
    assert not p.passed


# ---------- Fusion (all branches) ----------


def test_fusion_all_pass():
    live, reasons = decide_verdict(
        expired=False,
        passive_pass=True,
        challenge_done=True,
        tampering_suspected=False,
        spoof_suspected=False,
        multiple_faces=False,
        face_lost=False,
    )
    assert live and reasons == []


@pytest.mark.parametrize(
    "kwargs,expected_reason",
    [
        (dict(expired=True), "expired"),
        (dict(tampering_suspected=True), "frame_tampering_suspected"),
        (dict(multiple_faces=True), "multiple_faces"),
        (dict(face_lost=True), "face_lost"),
        (dict(spoof_suspected=True), "spoof_suspected"),
        (dict(passive_pass=False), "spoof_suspected"),
        (dict(challenge_done=False), "challenge_failed"),
    ],
)
def test_fusion_single_failure_branches(kwargs, expected_reason):
    base = dict(
        expired=False,
        passive_pass=True,
        challenge_done=True,
        tampering_suspected=False,
        spoof_suspected=False,
        multiple_faces=False,
        face_lost=False,
    )
    base.update(kwargs)
    live, reasons = decide_verdict(**base)
    assert not live
    assert expected_reason in reasons


def test_fusion_face_lost_suppresses_redundant_spoof_reason():
    live, reasons = decide_verdict(
        expired=False,
        passive_pass=False,  # unavoidable when the face was never seen
        challenge_done=False,
        tampering_suspected=False,
        spoof_suspected=False,
        multiple_faces=False,
        face_lost=True,
    )
    assert not live
    assert "face_lost" in reasons and "spoof_suspected" not in reasons


# ---------- HTTP endpoints (no models needed) ----------


def _client():
    from fastapi.testclient import TestClient
    from app.main import app

    return TestClient(app)


API_KEY = "my-secret-api-key"


def test_create_session_requires_api_key():
    r = _client().post("/api/v1/liveness-sessions")
    assert r.status_code in (401, 403)


def test_create_session_and_pending_result():
    client = _client()
    r = client.post(f"/api/v1/liveness-sessions?api_key={API_KEY}")
    assert r.status_code == 200
    data = r.json()
    assert data["challenge"] in ("blink", "turn_left", "turn_right")
    if data["challenge"] == "blink":
        assert data["required_blinks"] in (2, 3)
    assert data["stream_token"]
    assert data["stream_path"].endswith(f"/{data['session_id']}/stream")

    res = client.get(f"/api/v1/liveness-sessions/{data['session_id']}/result?api_key={API_KEY}")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "pending"
    assert body["live"] is None


def test_result_unknown_session_404():
    r = _client().get(f"/api/v1/liveness-sessions/doesnotexist/result?api_key={API_KEY}")
    assert r.status_code == 404


def test_websocket_rejects_bad_token():
    from starlette.websockets import WebSocketDisconnect as ClientDisconnect

    client = _client()
    created = client.post(f"/api/v1/liveness-sessions?api_key={API_KEY}").json()
    with pytest.raises(ClientDisconnect):
        with client.websocket_connect(
            f"/api/v1/liveness-sessions/{created['session_id']}/stream?token=wrong"
        ) as ws:
            ws.receive_json()


def test_websocket_sequence_replay_terminates_session(monkeypatch):
    """Streaming out-of-order sequence numbers ends the session with a tampering verdict (models mocked)."""
    from app.liveness_session import router as ls_router
    from app.liveness_session.vision import FrameAnalysis

    monkeypatch.setattr(
        ls_router,
        "_analyze",
        lambda jpeg, run_passive: FrameAnalysis(
            face_count=1, bbox=(0, 0, 10, 10), ear=0.3, yaw_degrees=0.0
        ),
    )
    client = _client()
    created = client.post(f"/api/v1/liveness-sessions?api_key={API_KEY}").json()
    jpeg = b"\xff\xd8\xff" + b"x" * 10

    with client.websocket_connect(
        f"/api/v1/liveness-sessions/{created['session_id']}/stream?token={created['stream_token']}"
    ) as ws:
        ws.send_bytes((2).to_bytes(4, "big") + jpeg + b"a")
        ws.receive_json()  # progress
        ws.send_bytes((1).to_bytes(4, "big") + jpeg + b"b")  # out of order -> terminate
        final = ws.receive_json()
        assert final["type"] == "result"
        assert final["live"] is False
        assert "frame_tampering_suspected" in final["reasons"]

    res = client.get(
        f"/api/v1/liveness-sessions/{created['session_id']}/result?api_key={API_KEY}"
    ).json()
    assert res["status"] == "done"
    assert res["live"] is False


def test_websocket_static_image_loop_fails_with_reason(monkeypatch):
    """Acceptance check: looping a static image -> live=false with a specific reason (models mocked)."""
    from app.liveness_session import router as ls_router
    from app.liveness_session.vision import FrameAnalysis

    monkeypatch.setattr(
        ls_router,
        "_analyze",
        lambda jpeg, run_passive: FrameAnalysis(
            face_count=1, bbox=(0, 0, 10, 10), ear=0.3, yaw_degrees=0.0
        ),
    )
    client = _client()
    created = client.post(f"/api/v1/liveness-sessions?api_key={API_KEY}").json()
    jpeg = b"\xff\xd8\xff" + b"static"

    with client.websocket_connect(
        f"/api/v1/liveness-sessions/{created['session_id']}/stream?token={created['stream_token']}"
    ) as ws:
        final = None
        for i in range(10):  # identical frames -> duplicate-frame tampering gate
            ws.send_bytes((i + 1).to_bytes(4, "big") + jpeg)
            msg = ws.receive_json()
            if msg["type"] == "result":
                final = msg
                break
        assert final is not None
        assert final["live"] is False
        assert "frame_tampering_suspected" in final["reasons"]
