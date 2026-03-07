#!/usr/bin/env python3
"""
Capture a live frame from the camera and send it to the Liveness API.
Use this to get live=true: the image is a real capture, not a saved photo.

Requirements: opencv-python (pip install opencv-python)
Usage: python scripts/capture_and_test_live.py [API_BASE_URL]
Default URL: http://localhost:8000
"""
import base64
import json
import sys
import urllib.request

try:
    import cv2
except ImportError:
    print("Install opencv-python: pip install opencv-python")
    sys.exit(1)

BASE_URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
API_URL = f"{BASE_URL}/api/v1/liveness"

def main():
    print("Opening camera... (look at the camera, then press SPACE to capture, or Q to quit)")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open camera.")
        sys.exit(1)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Press SPACE to capture (live test)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" "):
            # Encode as JPEG, then base64
            _, buf = cv2.imencode(".jpg", frame)
            b64 = base64.b64encode(buf.tobytes()).decode("ascii")
            cap.release()
            cv2.destroyAllWindows()
            print("Sending to API...")
            req = urllib.request.Request(
                API_URL,
                data=json.dumps({"image_base64": b64}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=30) as r:
                    out = json.loads(r.read().decode())
                    print(json.dumps(out, indent=2))
                    print("\n-- live:", out.get("live"), "| confidence:", out.get("confidence"))
            except Exception as e:
                print("Request failed:", e)
            return
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
