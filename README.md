# Liveness API – Request / Response

Simple HTTP API for face liveness:

- **Single-frame liveness**: `POST /api/v1/liveness`
- **Motion-based liveness**: `POST /api/v1/liveness-motion`
- **Health**: `GET /api/health`

All examples below assume the API is reachable at `http://localhost:8082` (adjust host/port as needed).

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

---

## Single-frame liveness

**Endpoint**

```http
POST /api/v1/liveness
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

**Endpoint**

```http
POST /api/v1/liveness-motion
Content-Type: application/json
```

**Request body**

```json
{
  "frames": [
    "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQ...",   // frame 1
    "/9j/4AAQSkZJRgABAQ...",                          // frame 2 (raw base64)
    "...optional more frames..."
  ]
}
```

Minimum 2 frames are required. Frames should be captured in sequence with slight head movement between them.

**Response body** (example):

```json
{
  "live": true,
  "confidence": 0.91,
  "details": {
    "frame_count": 2,
    "motion_ok": true,
    "motion_max_shift_ratio": 0.015,
    "per_frame": [
      {
        "live": true,
        "confidence": 0.93,
        "details": {
          "face_count": 1,
          "bbox": [90, 210, 290, 340]
        }
      },
      {
        "live": true,
        "confidence": 0.91,
        "details": {
          "face_count": 1,
          "bbox": [95, 215, 295, 345]
        }
      }
    ]
  },
  "errors": []
}
```

- `live`: `true` only if **all frames are live** and **enough motion** is detected between frames.
- `motion_max_shift_ratio`: approximate maximum head movement between frames as a fraction of image size.

---

## Error responses

Common error shapes:

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
