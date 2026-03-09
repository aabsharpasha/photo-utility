#!/usr/bin/env sh
# Quick test of the Liveness API (run with API at http://localhost:8000).
# Usage: ./scripts/test_liveness.sh [BASE_URL]
# With image (single-frame): IMAGE_PATH=./photo.jpg ./scripts/test_liveness.sh
# With motion (two frames): IMAGE_PATH=./frame1.jpg MOTION_IMAGE_PATH=./frame2.jpg ./scripts/test_liveness.sh
set -e
BASE_URL="${1:-http://localhost:8000}"
API="${BASE_URL}/api"

echo "=== Health ==="
curl -s "${API}/health" | python3 -m json.tool

echo ""
echo "=== Liveness (minimal base64 - expect no face / low confidence) ==="
curl -s -X POST "${API}/v1/liveness" \
  -H "Content-Type: application/json" \
  -d '{"image_base64":"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="}' \
  | python3 -m json.tool

if [ -n "$IMAGE_PATH" ] && [ -f "$IMAGE_PATH" ]; then
  echo ""
  echo "=== Liveness (from file: $IMAGE_PATH) ==="
  B64=$(python3 -c "import base64,sys; print(base64.b64encode(open(sys.argv[1],'rb').read()).decode())" "$IMAGE_PATH")
  curl -s -X POST "${API}/v1/liveness" \
    -H "Content-Type: application/json" \
    -d "{\"image_base64\":\"$B64\"}" \
    | python3 -m json.tool
fi

if [ -n "$IMAGE_PATH" ] && [ -n "$MOTION_IMAGE_PATH" ] && [ -f "$IMAGE_PATH" ] && [ -f "$MOTION_IMAGE_PATH" ]; then
  echo ""
  echo "=== Motion liveness (from files: $IMAGE_PATH, $MOTION_IMAGE_PATH) ==="
  B64_1=$(python3 -c "import base64,sys; print(base64.b64encode(open(sys.argv[1],'rb').read()).decode())" "$IMAGE_PATH")
  B64_2=$(python3 -c "import base64,sys; print(base64.b64encode(open(sys.argv[1],'rb').read()).decode())" "$MOTION_IMAGE_PATH")
  curl -s -X POST "${API}/v1/liveness-motion" \
    -H "Content-Type: application/json" \
    -d "{\"frames\":[\"$B64_1\",\"$B64_2\"]}" \
    | python3 -m json.tool
fi
