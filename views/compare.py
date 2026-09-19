"""Compare page: run several models on the same image."""
import hashlib
import io

import pandas as pd
import plotly.express as px
import streamlit as st

from src import config, database, models, ui


def render() -> None:
    ui.hero("Compare models", "Run several networks on the same image and see how they differ.", "⚖️")

    names = models.available_models()
    defaults = [n for n in ("MobileNetV2", "ResNet18", "EfficientNet-B0") if n in names]
    chosen = st.multiselect("Models to compare", names, default=defaults)
    save = st.toggle("Save these results to history", value=False)
    file = st.file_uploader("Image to classify", type=config.ALLOWED_TYPES, key="compare_file")

    if file is None:
        ui.empty_state("No image yet", "Upload one image and every selected model will classify it.", "⚖️")
        return
    if len(chosen) < 2:
        st.info("Select at least two models to compare.")
        return

    raw = file.getvalue()
    digest = hashlib.md5(raw).hexdigest()[:10]
    image = models.load_image(io.BytesIO(raw))

    results = []
    bar = st.progress(0.0, text="Starting...")
    for i, name in enumerate(chosen):
        bar.progress(i / len(chosen), text=f"Running {name} (the first run downloads its weights)...")
        loaded = models.load_model(name)
        pred = models.predict(loaded, image, top_k=3, warmup=True)
        results.append((name, loaded, pred))
    bar.empty()

    logged = st.session_state.setdefault("logged_keys", set())
    if save:
        for name, _, pred in results:
            key = f"{digest}:{name}:compare"
            if key not in logged:
                database.add_prediction(image, file.name, name, pred, "compare")
                logged.add(key)

    rows = [{
        "Model": name,
        "Prediction": pred["label"],
        "Confidence (%)": round(pred["confidence"] * 100, 1),
        "Speed (ms)": round(pred["latency_ms"], 1),
        "Parameters (M)": round(loaded.params_m, 1),
    } for name, loaded, pred in results]
    table = pd.DataFrame(rows)

    answers = set(table["Prediction"].str.lower())
    best = table.loc[table["Confidence (%)"].idxmax()]
    fastest = table.loc[table["Speed (ms)"].idxmin()]
    ui.kpi_row([
        ("Agreement", "All models agree" if len(answers) == 1 else f"{len(answers)} different answers"),
        ("Most confident", best["Model"], f"{best['Confidence (%)']:.1f}% for {best['Prediction']}"),
        ("Fastest", fastest["Model"], f"{fastest['Speed (ms)']:.0f} ms per image"),
    ])
    st.write("")

    left, right = st.columns([1, 2.2], gap="large")
    left.image(image, caption=file.name)
    with right:
        st.dataframe(table, hide_index=True, column_config={
            "Confidence (%)": st.column_config.ProgressColumn("Confidence", min_value=0, max_value=100, format="%.1f%%"),
        })

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Confidence**")
        fig = px.bar(table, x="Model", y="Confidence (%)", color="Model", color_discrete_sequence=ui.PALETTE)
        fig.update_layout(showlegend=False)
        fig.update_yaxes(range=[0, 100], title="")
        fig.update_xaxes(title="")
        ui.show_chart(ui.style_fig(fig, 300), "cmp_conf")
    with c2:
        st.markdown("**Speed (lower is faster)**")
        fig = px.bar(table, x="Model", y="Speed (ms)", color="Model", color_discrete_sequence=ui.PALETTE)
        fig.update_layout(showlegend=False)
        fig.update_yaxes(title="ms")
        fig.update_xaxes(title="")
        ui.show_chart(ui.style_fig(fig, 300), "cmp_speed")

    st.markdown("**Top 3 answers from each model**")
    cols = st.columns(len(results))
    for col, (name, _, pred) in zip(cols, results):
        with col.container(border=True):
            st.markdown(f"**{name}**")
            for label, prob in pred["top_k"]:
                st.write(f"{label}  \n`{prob * 100:.1f}%`")
