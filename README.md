# Liveness API – Request / Response

Simple HTTP API for face liveness and face match:

- **Single-frame liveness**: `POST /api/v1/liveness`
- **Motion-based liveness**: `POST /api/v1/liveness-motion`
- **Face match (Rekognition-style)**: `POST /api/v1/face-match`
- **Health**: `GET /api/health` · **Readiness (loads models)**: `GET /api/ready`

All examples below assume the API is reachable at `http://localhost:8082` (adjust host/port as needed).

## Authentication

All liveness and face-match endpoints require an API key as a **query parameter**:

```http
POST /api/v1/liveness-motion?api_key=<API_KEY_QUERY_VALUE>
```

Missing/invalid key → `401 {"detail": "Invalid or missing API key"}`. Health/ready endpoints are open.

---

## Health

**Request**

```http
GET /api/health
```

**Response (200)** – example:

```json
{
  "status": "ok",
  "version": "1.0.0",
  "environment": "production"
}
```

`GET /api/ready` additionally triggers model load and returns `503 {"status": "degraded"}` until models are available.

---

## Single-frame liveness

**Endpoint**

```http
POST /api/v1/liveness?api_key=...
Content-Type: application/json
```

**Request body**

```json
{
  "image_base64": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQ..."
}
```

`image_base64` may be **raw base64** (`/9j/4AA...`) or a **data URL** (`data:image/...;base64,...`).

**Response body** (example):

```json
{
  "live": true,
  "confidence": 0.92,
  "details": {
    "laplacian_variance": 45.3,
    "sharpness_score": 1.0,
    "face_count": 1,
    "detection_backend": "insightface",
    "best_det_score": 0.88,
    "bbox": [100, 200, 300, 350],
    "largest_face_area_ratio": 0.12,
    "heuristic_confidence": 0.90,
    "antispoof_real_score": 0.97,
    "antispoof_spoof_score": 0.03,
    "antispoof": "enabled",
    "reason": "OK"
  },
  "errors": []
}
```

- `live`: `true` if the frame is considered live.
- `confidence`: combined liveness confidence in `[0, 1]`.
- `details`: diagnostic fields (can change over time).
- `errors`: non-empty only if something went wrong.

---

## Motion-based liveness (multiple frames)

Multi-frame liveness tuned to reject presentation attacks (screen/video replay, held prints). The user
moves their head naturally between captures — **any direction**; there is no prescribed pattern.

**Endpoint**

```http
POST /api/v1/liveness-motion?api_key=...
Content-Type: application/json
```

**Request body**

```json
{
  "frames": [
    "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQ...",
    "/9j/4AAQSkZJRgABAQ...",
    "/9j/4AAQSkZJRgABAQ..."
  ]
}
```

Minimum **3 frames** by default (`motion_min_frames`). Recommended capture strategy per frame:
prompt a head movement, let the user settle and hold, then capture a **sharp** (non-blurred) frame.
Sending motion-blurred transitional frames is the most common cause of false rejects.

**`live` is `true` only when ALL of these gates pass:**

| Gate | Meaning |
| --- | --- |
| `per_frame_liveness` | Enough frames pass face detection + sharpness/size + anti-spoof (strict quorum, or relaxed quorum when enabled) |
| `head_motion` | Face center moves enough between frames (`motion_min_normalized_shift`, any direction) |
| `face_identity_continuity` | Consecutive frames contain the **same face** (embedding similarity) |
| `single_face_per_frame` | Exactly one face in every frame |
| `moire_gate` (optional) | No screen-like periodic pattern (FFT moiré heuristic) |

**Response body** (example):

```json
{
  "live": true,
  "confidence": 0.91,
  "details": {
    "frame_count": 3,
    "motion_ok": true,
    "motion_pair_shift_ratios": [0.031, 0.052],
    "motion_max_shift_ratio": 0.052,
    "motion_min_normalized_shift": 0.02,
    "identity_ok": true,
    "single_face_ok": true,
    "per_frame_face_counts": [1, 1, 1],
    "consecutive_face_similarities": [99.1, 98.7],
    "frames_live_count": 3,
    "aggregate_confidence": 0.91,
    "replay_metrics": { "moire_scores": [0.0, 0.0, 0.0], "moire_max": 0.0 },
    "moire_gate_enabled": true,
    "moire_gate_ok": true,
    "live_rejection_reasons": [],
    "live_gate_summary": {
      "passed": [
        "per_frame_liveness",
        "head_motion",
        "face_identity_continuity",
        "single_face_per_frame",
        "moire_gate"
      ],
      "failed": []
    },
    "live_mismatch_explanation": null,
    "per_frame": [
      {
        "live": true,
        "confidence": 0.93,
        "details": {
          "face_count": 1,
          "bbox": [90, 210, 290, 340],
          "reason": "OK",
          "antispoof_real_score": 0.97,
          "antispoof_context_real_score": 0.41
        }
      }
    ]
  },
  "errors": []
}
```

- `confidence`: blended per-frame confidence (`details.aggregate_confidence`); can be high while
  `live` is `false` if an auxiliary gate failed — check `live_gate_summary` / `live_rejection_reasons`.
- `live_mismatch_explanation`: human-readable reason whenever `live` is `false`.
- Tuning knobs live in `app/config.py` (`motion_*` fields): anti-spoof thresholds, moiré gate,
  per-frame quorum, minimum shift, identity similarity.

---

## Face match (Rekognition-style)

Compare a source face against faces in a target image. Request/response follow AWS Rekognition
`CompareFaces` (subset).

**Endpoint**

```http
POST /api/v1/face-match?api_key=...
Content-Type: application/json
```

**Request body**

```json
{
  "SourceImage": { "Bytes": "data:image/jpeg;base64,/9j/..." },
  "TargetImage": { "Bytes": "/9j/..." },
  "SimilarityThreshold": 45
}
```

**Response body** (example):

```json
{
  "Match": true,
  "SourceImageFace": { "BoundingBox": {}, "Confidence": 99.2 },
  "FaceMatches": [
    { "Similarity": 97.4, "Face": { "BoundingBox": {}, "Confidence": 98.8 } }
  ],
  "UnmatchedFaces": []
}
```

---

## Session-based liveness (challenge–response streaming)

Single-use sessions with a server-chosen random challenge, verified over a frame-streaming
WebSocket. Endpoints are named `liveness-sessions` to avoid clashing with the existing
`/api/v1/liveness*` routes.

| Endpoint | Purpose |
| --- | --- |
| `POST /api/v1/liveness-sessions?api_key=...` | Create session, returns challenge + short-lived stream token |
| `WS /api/v1/liveness-sessions/{id}/stream?token=...` | Client streams JPEG frames; server runs checks and pushes progress |
| `GET /api/v1/liveness-sessions/{id}/result?api_key=...` | Idempotent backend-to-backend verdict fetch (never trust the client to relay it) |

**Create session** → `200`:

```json
{
  "session_id": "6f1c...",
  "stream_token": "kJ3v...",
  "challenge": "blink",
  "required_blinks": 2,
  "expires_in_seconds": 20,
  "stream_path": "/api/v1/liveness-sessions/6f1c.../stream"
}
```

Challenges: `blink` (random 2–3 blinks), `turn_left`, `turn_right` (hold the turn, then
**return to frontal** — mandatory, defeats side-profile photos).

**WebSocket protocol**: client sends binary messages, each a 4-byte big-endian sequence
number followed by JPEG bytes (5–10 fps, max 500 KB/frame). Out-of-order or duplicate
sequence numbers terminate the session (replay defense). Server pushes JSON `progress`
messages (`hint`, blink/turn state, passive status) and a final `result` message.

**Verdict**: `live = passive anti-spoof pass AND challenge done AND no anti-gaming failure
AND not expired`. Failure reasons: `expired | challenge_failed | spoof_suspected |
multiple_faces | face_lost | frame_tampering_suspected`. Verdicts are stored 24 h
(in-memory by default; set `REDIS_URL` for Redis).

**Config**: every threshold is an env var with defaults in `app/config.py`
(`LIVENESS_SESSION_*`, see `.env.example`).

**Demo client**: `python examples/webcam_client.py --base http://localhost:8082 --api-key ...`
(requires `pip install opencv-python websockets requests`).

**Known limitation**: server-side checks cannot fully defeat virtual-camera injection
(OBS/deepfake feeds). Implemented mitigations: sequence-number replay rejection, duplicate
frame hashing, face-box continuity (IoU), moiré screen detection, passive MiniFAS anti-spoof.
Roadmap: signed client capture attestation and a screen-flash color-reflection challenge.

---

## Error responses

Common error shapes:

- Missing/invalid API key:

```json
{
  "detail": "Invalid or missing API key"
}
```

- Invalid/too large image:

```json
{
  "detail": "Invalid or unsupported image_base64"
}
```

- Payload too big:

```json
{
  "detail": "Image payload exceeds max size (10485760 bytes)"
}
```

- Not enough frames (motion):

```json
{
  "detail": "At least 3 frames are required for motion liveness"
}
```

Standard FastAPI validation errors (missing fields, wrong types) are returned in the usual:

```json
{
  "detail": [
    {
      "loc": ["body", "image_base64"],
      "msg": "Field required",
      "type": "value_error.missing"
    }
  ]
}
```
