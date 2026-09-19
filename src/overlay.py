"""Draws the prediction panel on top of a camera frame (OpenCV, BGR colours)."""
from __future__ import annotations

import cv2

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_GREEN, _AMBER, _RED = (90, 158, 30), (11, 138, 217), (80, 69, 214)  # BGR


def draw_overlay(frame, results, fps: float, model_name: str):
    """results = [(label, probability), ...]. Returns a new annotated frame."""
    h, w = frame.shape[:2]
    panel_w = min(w - 20, 400)
    panel_h = 48 + 34 * len(results)

    shade = frame.copy()
    cv2.rectangle(shade, (10, 10), (10 + panel_w, 10 + panel_h), (45, 27, 15), -1)
    out = cv2.addWeighted(shade, 0.72, frame, 0.28, 0)

    cv2.putText(out, f"{model_name}   {fps:4.1f} FPS", (22, 35), _FONT, 0.52,
                (235, 225, 210), 1, cv2.LINE_AA)
    for i, (label, prob) in enumerate(results):
        y = 54 + i * 34
        full = panel_w - 24
        cv2.rectangle(out, (22, y), (22 + full, y + 24), (78, 60, 44), -1)
        color = _GREEN if prob >= 0.7 else _AMBER if prob >= 0.4 else _RED
        cv2.rectangle(out, (22, y), (22 + int(full * prob), y + 24), color, -1)
        cv2.putText(out, f"{label[:24]}  {prob * 100:4.1f}%", (30, y + 17), _FONT, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
    return out
