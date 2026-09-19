"""Home page: overview, quick stats and where to start."""
import streamlit as st

from src import database, ui

FEATURES = [
    ("🖼️", "Classify images", "Drop in one or many pictures and get the predicted category with a confidence score."),
    ("📊", "Results dashboard", "Charts of every prediction: activity, top categories, confidence and speed per model."),
    ("🕘", "History", "Every result is saved with a thumbnail. Search, filter, export to CSV or delete."),
    ("🧪", "Model evaluation", "Accuracy, precision, recall, F1 and a confusion matrix on your own test folder."),
    ("🏋️", "Train on your data", "Teach a CNN your own categories with transfer learning, no code needed."),
    ("📷", "Live camera", "Classify from your webcam, as a snapshot or as a live stream with an on-screen overlay."),
    ("⚖️", "Compare models", "Run several networks on the same image and compare answers, confidence and speed."),
]


def render() -> None:
    ui.hero("Image classification studio",
            "Upload a picture, see what the model thinks and how sure it is. "
            "Train your own categories, evaluate them and compare models.", "🧠")

    df = database.fetch_history()
    if df.empty:
        ui.kpi_row([("Predictions", "0", "Nothing classified yet"), ("Average confidence", "–"),
                    ("Average speed", "–"), ("Categories seen", "–")])
    else:
        ui.kpi_row([
            ("Predictions", f"{len(df):,}", f"Latest: {df['created_at'].iloc[0]:%d %b %Y, %H:%M}"),
            ("Average confidence", f"{df['confidence_pct'].mean():.1f}%"),
            ("Average speed", f"{df['latency_ms'].mean():.0f} ms", "per image"),
            ("Categories seen", f"{df['label'].nunique()}"),
        ])

    st.write("")
    st.subheader("What you can do here")
    for start in range(0, len(FEATURES), 3):
        cols = st.columns(3)
        for col, (icon, title, text) in zip(cols, FEATURES[start:start + 3]):
            col.markdown(ui.feature_card(title, text, icon), unsafe_allow_html=True)

    st.subheader("Getting started")
    st.markdown(
        "1. Open **Classify** and drop in a photo. The built-in models recognise 1,000 everyday categories.\n"
        "2. Want your own categories? Open **Train** and point it at a folder of images (one sub-folder per category).\n"
        "3. Open **Evaluate** to measure the new model, then **Compare** to see how it stacks up against the others."
    )
    ui.md('<div class="chips"><span>PyTorch</span><span>CNN</span><span>Transfer learning</span>'
          '<span>Computer vision</span><span>Streamlit</span><span>SQLite</span></div>')
