#!/usr/bin/env sh
# Quick test of the Liveness API (run with API at http://localhost:8000).
# Usage: ./scripts/test_liveness.sh [BASE_URL]
# With image: IMAGE_PATH=./photo.jpg ./scripts/test_liveness.sh
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
