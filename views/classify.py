"""Classify page: upload images, get predictions with confidence."""
import hashlib
import io

import pandas as pd
import streamlit as st

from src import config, database, models, ui


def render() -> None:
    ui.hero("Classify images",
            "Upload one or more images. Each one is classified and shown with its confidence.", "🖼️")

    c1, c2, c3 = st.columns([2, 1.2, 1])
    model_name = c1.selectbox("Model", models.available_models(),
                              help="Pretrained models know 1,000 ImageNet categories. Models you train appear as 'Custom'.")
    top_k = c2.slider("Predictions to show", 1, 10, 5)
    save = c3.toggle("Save to history", value=True)

    files = st.file_uploader("Drag and drop images here, or browse", type=config.ALLOWED_TYPES,
                             accept_multiple_files=True)
    if not files:
        ui.empty_state("No image yet", "Upload a JPG, PNG, WebP or BMP file to see a prediction.", "⬆️")
        return

    with st.spinner("Loading model (the first run downloads the weights)..."):
        loaded = models.load_model(model_name)

    logged = st.session_state.setdefault("logged_keys", set())
    rows = []
    for idx, file in enumerate(files):
        data = file.getvalue()
        digest = hashlib.md5(data).hexdigest()[:10]
        try:
            image = models.load_image(io.BytesIO(data))
        except Exception:
            st.error(f"**{file.name}** could not be opened as an image. Try a different file.")
            continue

        pred = models.predict(loaded, image, top_k)
        history_key = f"{digest}:{model_name}"
        if save and history_key not in logged:
            database.add_prediction(image, file.name, model_name, pred, "upload")
            logged.add(history_key)

        st.markdown(f"#### {file.name}")
        ui.show_prediction(image, pred, key=f"{idx}_{digest}", model=model_name)
        st.divider()
        rows.append({"file": file.name, "prediction": pred["label"],
                     "confidence_pct": round(pred["confidence"] * 100, 2),
                     "latency_ms": round(pred["latency_ms"], 1), "model": model_name})

    if len(rows) > 1:
        st.subheader("Batch summary")
        table = pd.DataFrame(rows)
        st.dataframe(table, hide_index=True)
        st.download_button("Download results (CSV)", table.to_csv(index=False).encode("utf-8"),
                           file_name="predictions.csv", mime="text/csv")
