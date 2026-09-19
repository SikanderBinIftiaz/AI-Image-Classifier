"""Real-time webcam classification in its own window (fast, smooth).

    python realtime_camera.py                       # default: MobileNetV2, camera 0
    python realtime_camera.py --model ResNet18
    python realtime_camera.py --model "Custom: my_classifier"

Keys:  q or Esc = quit    m = next model    s = save a screenshot
"""
import argparse
import sys
import time
from datetime import datetime

import cv2
from PIL import Image

from src import config, models
from src.overlay import draw_overlay


def main() -> None:
    parser = argparse.ArgumentParser(description="Real-time camera classification")
    parser.add_argument("--model", default="MobileNetV2", help="model name (see the app's model list)")
    parser.add_argument("--camera", type=int, default=0, help="camera number (0 = default webcam)")
    parser.add_argument("--top-k", type=int, default=3, help="predictions shown on screen")
    parser.add_argument("--every", type=int, default=2, help="analyse every N-th frame")
    args = parser.parse_args()

    names = models.available_models()
    if args.model not in names:
        sys.exit("Unknown model. Choose one of:\n  " + "\n  ".join(names))
    index = names.index(args.model)

    print(f"Loading {names[index]} on {models.DEVICE} ...")
    loaded = models.load_model(names[index])

    capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        sys.exit("Could not open the camera. Close other apps using it or try --camera 1.")

    print("Running.  q/Esc = quit   m = next model   s = screenshot")
    smoothed, frame_id, fps, last = None, 0, 0.0, time.perf_counter()
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)

            if frame_id % args.every == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                probs, _ = models.predict_probs(loaded, Image.fromarray(rgb))
                # Blend with the previous frames so the label does not flicker.
                smoothed = probs if smoothed is None else 0.6 * smoothed + 0.4 * probs
            frame_id += 1

            now = time.perf_counter()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - last, 1e-3))
            last = now

            results = models.top_k_from_probs(smoothed, loaded.classes, args.top_k)
            shown = draw_overlay(frame, results, fps, names[index])
            cv2.imshow("VisionAI - live detection", shown)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("m"):
                index = (index + 1) % len(names)
                print(f"Switching to {names[index]} ...")
                loaded = models.load_model(names[index])
                smoothed = None
            elif key == ord("s"):
                path = config.SCREENSHOT_DIR / f"capture_{datetime.now():%Y%m%d_%H%M%S}.jpg"
                cv2.imwrite(str(path), shown)
                print(f"Saved {path}")
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
