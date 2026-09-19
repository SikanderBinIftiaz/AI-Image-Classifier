"""VisionAI - Image classification studio.  Start with:  streamlit run app.py"""
import streamlit as st

st.set_page_config(
    page_title="VisionAI · Image classification studio",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

from src import config, database, models, ui  # noqa: E402
from views import camera, classify, compare, dashboard, evaluate, history, home, train  # noqa: E402

database.init_db()
ui.inject_css()

PAGES = {
    "🏠  Home": home.render,
    "🖼️  Classify": classify.render,
    "📊  Dashboard": dashboard.render,
    "🕘  History": history.render,
    "🧪  Evaluate": evaluate.render,
    "🏋️  Train": train.render,
    "📷  Camera": camera.render,
    "⚖️  Compare": compare.render,
}

with st.sidebar:
    ui.brand(config.APP_NAME, config.APP_TAGLINE)
    choice = st.radio("Navigation", list(PAGES), label_visibility="collapsed")
    st.divider()
    device = "GPU (CUDA)" if models.DEVICE.type == "cuda" else "CPU"
    custom_count = len(models.list_custom_models())
    st.markdown(
        f'<div class="side-note">Running on {device}<br>'
        f'{len(models.MODEL_ZOO)} pretrained models · {custom_count} custom</div>',
        unsafe_allow_html=True,
    )

PAGES[choice]()
