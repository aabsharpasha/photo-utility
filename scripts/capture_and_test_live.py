#!/usr/bin/env python3
"""
Capture frames from the camera and send them to the Liveness API.

Default mode: motion-based liveness
- Captures 2 frames with slight head movement between them
- Calls POST /api/v1/liveness-motion with { frames: [frame1, frame2] }

Requirements: opencv-python (pip install opencv-python)
Usage: python scripts/capture_and_test_live.py [API_BASE_URL]
Default URL: http://localhost:8082 (docker-compose); use http://localhost:8000 for plain uvicorn/Docker.
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

BASE_URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8082"
SINGLE_URL = f"{BASE_URL}/api/v1/liveness"
MOTION_URL = f"{BASE_URL}/api/v1/liveness-motion"

def main():
    print("Opening camera...")
    print("Controls:")
    print("  SPACE  - motion-based liveness (2 frames; move head slightly)")
    print("  ENTER  - single-frame liveness (one capture)")
    print("  Q      - quit")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open camera.")
        sys.exit(1)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Press SPACE (motion) or ENTER (single) to capture", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" "):
            # Motion-based: capture 2 frames with slight head movement between them
            print("Capturing frame 1 (look at camera)...")
            ret1, frame1 = cap.read()
            if not ret1:
                print("Failed to capture frame 1.")
                continue
            cv2.imshow("Frame 1 captured - now move your head slightly", frame1)
            cv2.waitKey(500)
            print("Capturing frame 2 (after head movement)...")
            ret2, frame2 = cap.read()
            if not ret2:
                print("Failed to capture frame 2.")
                continue
            # Encode as JPEG, then base64
            _, buf1 = cv2.imencode(".jpg", frame1)
            _, buf2 = cv2.imencode(".jpg", frame2)
            b64_1 = base64.b64encode(buf1.tobytes()).decode("ascii")
            b64_2 = base64.b64encode(buf2.tobytes()).decode("ascii")
            cap.release()
            cv2.destroyAllWindows()
            print("Sending 2 frames to /api/v1/liveness-motion...")
            req = urllib.request.Request(
                MOTION_URL,
                data=json.dumps({"frames": [b64_1, b64_2]}).encode("utf-8"),
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
        if key == 13:  # ENTER key
            # Single-frame liveness
            print("Capturing single frame...")
            ret, frame_single = cap.read()
            if not ret:
                print("Failed to capture frame.")
                continue
            _, buf = cv2.imencode(".jpg", frame_single)
            b64 = base64.b64encode(buf.tobytes()).decode("ascii")
            cap.release()
            cv2.destroyAllWindows()
            print("Sending to /api/v1/liveness...")
            req = urllib.request.Request(
                SINGLE_URL,
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
