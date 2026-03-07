# Liveness API (InsightFace + Anti-spoof ONNX)

Enterprise FastAPI service for **face liveness check**: **InsightFace (Direct)** for detection, **MiniFAS-style ONNX** for anti-spoof, plus sharpness/size heuristics. **Docker-first, uv-based.**

## Features

- **InsightFace (Direct)**: RetinaFace `buffalo_l` for face detection.
- **Anti-spoof ONNX**: Optional MiniFAS-style model (128×128 RGB), e.g. [SuriAI/face-antispoof-onnx](https://github.com/SuriAI/face-antispoof-onnx).
- **Heuristics**: Laplacian sharpness, face area ratio, detection score.
- **Docker + uv**: No virtualenv; use Docker or `uv run` for local dev.
- **Enterprise**: Health/readiness, rate limiting, config via env, OpenAPI.

## Quick start (Docker)

```bash
cd liveness-api

# Build, then run (two commands work on all Docker/Compose setups)
docker compose build
docker compose up
```

If your Docker doesn’t support `docker compose up --build`, use the two-step form above or run `./scripts/docker-up.sh`. With standalone Compose: `docker-compose up --build`.

- **API**: http://localhost:8000  
- **Docs**: http://localhost:8000/docs  
- **Health**: `GET /api/health`  
- **Liveness**: `POST /api/v1/liveness` with `{"image_base64": "<base64 or data:image/...;base64,...>"}`  

InsightFace is downloaded at **build time** (no download on first request). Optional: place `antispoof.onnx` in `./models/` and mount it (see below).

**Container exits with code 137:** The process was killed (usually out-of-memory when loading models). Give the container more RAM: e.g. **Docker Desktop → Settings → Resources → Memory** set to at least **4GB**, or run with `docker compose --compatibility up` so the 2G limit in `docker-compose.yml` is applied.

### Anti-spoof model

- **Option A – Mount at run**: Put `antispoof.onnx` (e.g. from [SuriAI/face-antispoof-onnx](https://github.com/SuriAI/face-antispoof-onnx) `models/best_model.onnx`) in `./models/`. `docker-compose.yml` already mounts `./models` into `/app/models`.
- **Option B – Bake in image**: Build with a public URL:
  ```bash
  docker build --build-arg ANTISPOOF_MODEL_URL="https://.../best_model.onnx" -t liveness-api .
  ```
  (Use a real URL to the ONNX file; SuriAI may host it in Releases.)

Set `ANTISPOOF_ENABLED=false` to use only heuristics (no ONNX).

**Making anti-spoof work with real mobile selfies**

If real selfies are rejected as "Anti-spoof: classified as spoof", try (in order):

1. **Lower the threshold**  
   `ANTISPOOF_REAL_THRESHOLD=0.35` (default) or `0.25` for very lenient.

2. **Give more weight to heuristics**  
   `ANTISPOOF_WEIGHT=0.4` (default) so sharpness/face-size count more than the ONNX score.

3. **Flip real/spoof output index**  
   Some ONNX models output `[real, spoof]` instead of `[spoof, real]`. If your real selfies always get a low `antispoof_real_score` in the API response, try:
   - `ANTISPOOF_REAL_INDEX=0` (first class = real). Default is `1` (second class = real).

4. **Softer sharpness**  
   `LAPLACIAN_MIN=25` so mobile selfies aren’t rejected for blur before anti-spoof runs.

Example (Docker env or `.env`):

```bash
ANTISPOOF_ENABLED=true
ANTISPOOF_REAL_THRESHOLD=0.35
ANTISPOOF_WEIGHT=0.4
ANTISPOOF_REAL_INDEX=1
LAPLACIAN_MIN=25
```

**Important: Anti-spoof cannot tell "saved file" from "live camera".** The API only receives pixels (base64). It cannot know if the image was captured from a camera now or loaded from a file. MiniFAS models detect presentation attacks (printed photo or screen with reflections/moiré). A high-quality image from a file looks identical to a live capture, so a saved laptop photo can get antispoof_real_score 0.99 and pass. To reduce abuse: use only your app (camera-only, no gallery), or add motion-based liveness (blink/video) or challenge-response so a single static image cannot pass.

**Anti-spoof overhead when enabled:** One extra ONNX inference per liveness request (128×128 face crop). Typical cost: ~20–50 ms and ~1–2 MB RAM for the small MiniFAS model. Response includes `antispoof_real_score` and `antispoof_spoof_score` in `details`.

### Docker: “Bake / buildx isn’t installed” warning

If you see **“Docker Compose is configured to build using Bake, but buildx isn't installed”**, the build can still finish (Compose may fall back to the classic builder). To remove the warning and use the preferred builder:

- **Docker Desktop (Mac/Windows):** Buildx is usually included; ensure you’re on a recent version.
- **Linux:** Install the plugin: `docker buildx install` (or install the `docker-buildx-plugin` package for your distro).

**Alternative (no buildx):** build and run with plain Docker:

```bash
./scripts/docker-run-standalone.sh
```

Or manually: `docker build -t liveness-api:latest .` then  
`docker run -p 8000:8000 -e ENVIRONMENT=production -v insightface-cache:/app/.insightface liveness-api:latest`  
(Create the volume once with `docker volume create insightface-cache` if needed.)

## Local dev with uv (no Docker)

```bash
cd liveness-api
uv sync
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Requires [uv](https://github.com/astral-sh/uv) (`curl -LsSf https://astral.sh/uv/install.sh | sh`). No virtualenv needed; `uv run` uses the project env.

## API

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/health` | Liveness probe (no model) |
| GET | `/api/ready` | Readiness (loads models if needed) |
| POST | `/api/v1/liveness` | Check face liveness (rate limited) |
| POST | `/api/liveness` | Alias for `/api/v1/liveness` |

**Liveness response** (with anti-spoof enabled):

```json
{
  "live": true,
  "confidence": 0.87,
  "details": {
    "laplacian_variance": 250.5,
    "sharpness_score": 0.95,
    "face_count": 1,
    "detection_backend": "insightface",
    "best_det_score": 0.99,
    "bbox": [100, 80, 200, 240],
    "largest_face_area_ratio": 0.15,
    "heuristic_confidence": 0.82,
    "antispoof": "enabled",
    "antispoof_real_score": 0.96,
    "antispoof_spoof_score": 0.04,
    "reason": "OK"
  },
  "errors": []
}
```

## Testing the API

With the API running (e.g. `docker compose up`), use any of these:

**1. Swagger UI (easiest)**  
Open http://localhost:8000/docs, try `GET /api/health`, then `POST /api/v1/liveness` with a JSON body like:
```json
{"image_base64": "data:image/jpeg;base64,/9j/4AAQ..."}
```
Use “Try it out” and paste your base64 image (or a short dummy string to see the response shape).

**2. curl**
```bash
# Health
curl -s http://localhost:8000/api/health | python3 -m json.tool

# Liveness (tiny 1x1 PNG – will return no face / low confidence)
curl -s -X POST http://localhost:8000/api/v1/liveness \
  -H "Content-Type: application/json" \
  -d '{"image_base64":"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="}' \
  | python3 -m json.tool
```

**3. Test script (with optional image file)**  
```bash
./scripts/test_liveness.sh
# Or with your own photo:
IMAGE_PATH=./my_face.jpg ./scripts/test_liveness.sh
```

**4. Real live image (to get `live: true`)**  
The API returns `live: true` only when the image looks like a **live face** (real skin, not a print/screen). Use one of these:

- **Webcam capture (recommended):** Run the script that captures one frame from your camera and sends it to the API:
  ```bash
  pip install opencv-python   # if needed
  python scripts/capture_and_test_live.py
  ```
  Look at the camera, press **SPACE** to capture; the script POSTs that frame and prints the response.

- **Fresh selfie:** Take a **new** selfie with your phone (don’t use an old photo from gallery). Email or transfer it to your machine, then:
  ```bash
  IMAGE_PATH=./my_selfie.jpg ./scripts/test_liveness.sh
  ```

- **Swagger UI:** Open `/docs`, use your phone or webcam to take a selfie, convert it to base64 (e.g. use an online image-to-base64 tool or a small script), and paste into the `image_base64` field of `POST /api/v1/liveness`.

Using a **saved photo from disk**, a **screenshot**, or a **picture of a printed face** will often be classified as spoof (`live: false`). For `live: true`, the image must be a **live capture** (camera or just-taken selfie).

**5. Pytest (unit tests)**  
```bash
uv sync
uv run pytest tests/ -v
```
Requires the app to be importable (e.g. run from repo root). `tests/test_api.py` hits `/api/health` and `/api/v1/liveness` with invalid payloads.

## Log messages

- **`onnxruntime cpuid_info warning: Unknown CPU vendor`** – From ONNX Runtime’s C++ layer (e.g. in containers/ARM). Safe to ignore; inference still uses CPU. Newer ONNX Runtime versions may not print it.
- **`Applied providers: ['CPUExecutionProvider']`** / **`find model: ...`** – Normal: models are loading from `/app/.insightface`.

## Configuration

Env (see `.env.example`):

- **InsightFace**: `INSIGHTFACE_MODEL`, `INSIGHTFACE_DET_SIZE_W/H`, `INSIGHTFACE_CTX_ID`.
- **Liveness**: `LAPLACIAN_MIN`, `FACE_AREA_MIN_RATIO`, `LIVENESS_CONFIDENCE_THRESHOLD`, `MIN_DET_SCORE`.
- **Anti-spoof**: `ANTISPOOF_ENABLED`, `ANTISPOOF_MODEL_PATH`, `ANTISPOOF_INPUT_SIZE` (128), `ANTISPOOF_REAL_THRESHOLD`, `ANTISPOOF_WEIGHT`, `ANTISPOOF_CROP_PADDING_RATIO`, `ANTISPOOF_MIN_LOGIT_DIFF` (0.5 = require real_logit - spoof_logit ≥ 0.5 for pass). Preprocessing matches SuriAI (letterbox + BORDER_REFLECT_101). Output: index 0 = real, index 1 = spoof. Response includes `antispoof_logit_diff`, `antispoof_real_logit`, `antispoof_spoof_logit`.
- **Limits**: `MAX_IMAGE_SIZE_BYTES`, `RATE_LIMIT_PER_MINUTE`.

## Project layout

```
liveness-api/
├── app/
│   ├── config.py
│   ├── main.py
│   ├── api/routes.py, schemas.py
│   └── services/
│       ├── liveness.py   # InsightFace + heuristics + anti-spoof
│       └── antispoof.py   # ONNX anti-spoof (MiniFAS-style)
├── models/               # Mount or add antispoof.onnx here
├── pyproject.toml        # uv / pip deps
├── Dockerfile            # uv-based, multi-stage
├── docker-compose.yml
└── README.md
```

## License

Use under your project's license terms.
