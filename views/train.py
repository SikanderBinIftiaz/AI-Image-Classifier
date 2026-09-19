"""Train page: teach a CNN your own categories (custom dataset training)."""
import hashlib
import io
import zipfile

import pandas as pd
import plotly.express as px
import streamlit as st

from src import config, data, models, training, ui


def _dataset_source() -> str | None:
    choice = st.radio("Where are your images?",
                      ["Folder on this computer", "Upload a ZIP file", "Sample dataset (CIFAR-10 subset)"],
                      horizontal=True)
    if choice == "Folder on this computer":
        return st.text_input("Path to your dataset folder", placeholder=r"C:\datasets\pets   or   /home/me/pets",
                             help="One sub-folder per category, each containing images of that category.") or None
    if choice == "Upload a ZIP file":
        zip_file = st.file_uploader("ZIP file with one sub-folder per category", type=["zip"], key="train_zip")
        if zip_file is None:
            return None
        digest = hashlib.md5(zip_file.getvalue()).hexdigest()[:8]
        dest = config.DATASET_DIR / f"train_{digest}"
        if not dest.exists():
            with st.spinner("Extracting ZIP..."):
                try:
                    data.extract_zip(io.BytesIO(zip_file.getvalue()), dest)
                except (zipfile.BadZipFile, ValueError) as exc:
                    st.error(f"Could not read the ZIP file: {exc}")
                    return None
        return str(dest)
    sample = config.DATA_DIR / "sample_dataset" / "train"
    if not sample.exists():
        st.info("The sample dataset has not been created yet. Run  `python make_sample_dataset.py`  "
                "in a terminal, then come back.")
        return None
    return str(sample)


def _preview(path: str):
    """Show what was found in the dataset. Returns (root, counts) or None."""
    try:
        root, counts = data.scan_dataset(path)
    except FileNotFoundError as exc:
        st.error(str(exc))
        return None
    if len(counts) < 2:
        st.error("Found fewer than 2 category folders with images. Each category needs its own sub-folder.")
        return None
    ui.kpi_row([("Categories", f"{len(counts)}"), ("Images", f"{sum(counts.values()):,}"),
                ("Smallest category", f"{min(counts.values()):,}", min(counts, key=counts.get))])
    frame = pd.DataFrame({"category": list(counts), "images": list(counts.values())})
    fig = px.bar(frame, x="category", y="images", color_discrete_sequence=[ui.TEAL])
    fig.update_xaxes(title="")
    fig.update_yaxes(title="")
    ui.show_chart(ui.style_fig(fig, 260), "train_counts")
    if min(counts.values()) < 20:
        st.warning("Some categories have fewer than 20 images. Results improve a lot with 50 or more per category.")
    return root, counts


def _saved_models() -> None:
    st.subheader("Your trained models")
    names = models.list_custom_models()
    if not names:
        st.caption("Nothing here yet. Train a model above and it will be saved automatically.")
        return
    for name in names:
        meta = models.read_meta(name)
        with st.container(border=True):
            left, right = st.columns([4, 1])
            left.markdown(f"**{name}**  ·  {meta['arch']}  ·  {len(meta['classes'])} categories  ·  "
                          f"validation accuracy **{meta['best_val_acc'] * 100:.1f}%**")
            left.caption("Categories: " + ", ".join(meta["classes"][:12])
                         + (" ..." if len(meta["classes"]) > 12 else "") + f"  ·  trained {meta['created']}")
            if right.button("Delete", key=f"delmodel_{name}"):
                models.delete_custom_model(name)
                if st.session_state.get("last_training", {}).get("name") == name:
                    st.session_state.pop("last_training")
                st.rerun()


def render() -> None:
    ui.hero("Train on your own data",
            "Teach the network new categories. It starts from a pretrained CNN (transfer learning), "
            "so a few hundred images are often enough.", "🏋️")

    st.subheader("1. Choose your dataset")
    path = _dataset_source()
    scanned = _preview(path) if path else None

    st.subheader("2. Settings")
    c1, c2 = st.columns(2)
    name = c1.text_input("Name for your model", value="my_classifier")
    arch = c2.selectbox("Network", list(models.MODEL_ZOO),
                        help="MobileNetV2 is the fastest to train. ResNet50 is slower but often more accurate.")
    c1, c2, c3, c4 = st.columns(4)
    epochs = c1.slider("Epochs", 1, 30, 5, help="How many times the network sees the whole dataset.")
    batch = c2.select_slider("Batch size", options=[8, 16, 32, 64], value=32)
    lr = c3.select_slider("Learning rate", options=[0.0001, 0.0003, 0.001, 0.003], value=0.001)
    val_split = c4.slider("Validation share", 0.1, 0.4, 0.2, step=0.05,
                          help="Part of the images kept aside to check accuracy during training.")
    freeze = st.checkbox("Only train the final layer (faster, works well for small datasets)", value=True)
    st.caption(f"Training will run on: **{'GPU' if models.DEVICE.type == 'cuda' else 'CPU'}**"
               + ("" if models.DEVICE.type == "cuda" else " (slower; keep epochs low for a first try)"))

    st.subheader("3. Train")
    if st.button("Start training", type="primary", disabled=scanned is None):
        history: list[dict] = []
        bar = st.progress(0.0, text="Starting...")
        status = st.empty()
        chart = st.empty()

        def on_progress(event: dict) -> None:
            if event["type"] == "batch":
                part = event["fraction"] * 0.85 if event["phase"] == "train" else 0.85 + event["fraction"] * 0.15
                done = ((event["epoch"] - 1) + part) / event["epochs"]
                phase = "Learning" if event["phase"] == "train" else "Checking accuracy"
                bar.progress(min(done, 1.0), text=f"Epoch {event['epoch']} of {event['epochs']} · {phase}...")
            else:
                row = event["row"]
                history.append(row)
                status.markdown(f"Epoch {row['epoch']}: training accuracy **{row['train_acc'] * 100:.1f}%**, "
                                f"validation accuracy **{row['val_acc'] * 100:.1f}%**")
                ui.show_chart(ui.training_curves(history), f"live_{len(history)}", container=chart)

        try:
            meta = training.train_custom_model(
                scanned[0], name, arch=arch, epochs=epochs, batch_size=batch, lr=lr,
                val_split=val_split, freeze_backbone=freeze, progress_cb=on_progress)
            st.session_state["last_training"] = meta
            bar.progress(1.0, text="Training complete")
        except Exception as exc:
            st.error(f"Training stopped: {exc}")

    last = st.session_state.get("last_training")
    if last:
        st.success(f"Saved **{last['name']}** · best validation accuracy {last['best_val_acc'] * 100:.1f}%. "
                   "Choose it in Classify, Evaluate or Compare (listed as 'Custom').")

    st.divider()
    _saved_models()
