"""Camera page: classify a webcam snapshot, or run live detection."""
import hashlib
import io
import time

import streamlit as st
from PIL import Image

from src import database, models, ui


def _snapshot(model_name: str, top_k: int, save: bool) -> None:
    shot = st.camera_input("Point the camera at an object and take a picture")
    if shot is None:
        ui.empty_state("No picture yet", "Allow camera access in your browser, then press the camera button.", "📷")
        return
    raw = shot.getvalue()
    image = models.load_image(io.BytesIO(raw))
    loaded = models.load_model(model_name)
    pred = models.predict(loaded, image, top_k)

    digest = hashlib.md5(raw).hexdigest()[:10]
    logged = st.session_state.setdefault("logged_keys", set())
    key = f"{digest}:{model_name}:camera"
    if save and key not in logged:
        database.add_prediction(image, "camera snapshot", model_name, pred, "camera")
        logged.add(key)
    ui.show_prediction(image, pred, key=f"cam_{digest}", model=model_name)


def _live(model_name: str, top_k: int) -> None:
    c1, c2, c3 = st.columns([1, 1.4, 1.2])
    camera_index = int(c1.number_input("Camera number", min_value=0, max_value=5, value=0,
                                       help="0 is the default webcam. Try 1 for an external camera."))
    every = c2.select_slider("Analyse every N frames", options=[1, 2, 3, 5, 8], value=3,
                             help="Higher numbers are smoother on slow computers.")
    run = c3.toggle("Start live detection")

    if not run:
        ui.empty_state("Live detection is off",
                       "Turn on the switch above. The stream runs on the computer that hosts this app.", "🎥")
        return

    try:
        import cv2
        from src.overlay import draw_overlay
    except ImportError:
        st.error("OpenCV is not installed. Run  `pip install opencv-python`  and restart the app.")
        return

    loaded = models.load_model(model_name)
    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        st.error("Could not open the camera. Close other apps that use it, or try another camera number.")
        return

    slot = st.empty()
    smoothed, frame_id, fps, last = None, 0, 0.0, time.perf_counter()
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                st.warning("The camera stopped sending frames.")
                break
            frame = cv2.flip(frame, 1)  # mirror view feels natural
            if frame_id % every == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                probs, _ = models.predict_probs(loaded, Image.fromarray(rgb))
                smoothed = probs if smoothed is None else 0.6 * smoothed + 0.4 * probs
            frame_id += 1

            now = time.perf_counter()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - last, 1e-3))
            last = now

            results = models.top_k_from_probs(smoothed, loaded.classes, min(top_k, 3))
            annotated = draw_overlay(frame, results, fps, model_name)
            if annotated.shape[1] > 900:
                scale = 900 / annotated.shape[1]
                annotated = cv2.resize(annotated, None, fx=scale, fy=scale)
            slot.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB", output_format="JPEG")
    finally:
        capture.release()  # runs when you switch the toggle off


def render() -> None:
    ui.hero("Camera detection", "Classify what your webcam sees, as a snapshot or as a live stream.", "📷")

    c1, c2, c3 = st.columns([2, 1.2, 1])
    model_name = c1.selectbox("Model", models.available_models(), key="cam_model")
    top_k = c2.slider("Predictions to show", 1, 10, 3, key="cam_topk")
    save = c3.toggle("Save snapshots to history", value=False)

    # A radio (not tabs) so only one camera consumer is active at a time.
    mode = st.radio("Mode", ["Snapshot", "Live stream"], horizontal=True)
    if mode == "Snapshot":
        _snapshot(model_name, top_k, save)
    else:
        _live(model_name, top_k)
