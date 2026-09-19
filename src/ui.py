"""Look & feel: CSS theme and reusable components shared by every page."""
from __future__ import annotations

import html

import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

TEAL = "#0E8F80"
NAVY = "#1B2A41"
AMBER = "#D98A0B"
CORAL = "#D64550"
GREEN = "#1E9E5A"
GREY = "#AAB6C6"
PALETTE = [TEAL, NAVY, AMBER, CORAL, "#5B8DEF", GREY]

_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap');
:root{--ink:#0F1B2D;--muted:#5B6B82;--line:#DDE3EC;--surface:#FFFFFF;--teal:#0E8F80;--teal-soft:#E3F4F1;}
.stApp, .stApp p, .stApp label, .stApp li, .stApp td, .stApp th, .stApp button, .stApp input, .stApp textarea, .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5 {font-family:'Manrope',system-ui,-apple-system,'Segoe UI',sans-serif;}
.block-container{padding-top:1.8rem;padding-bottom:3rem;max-width:1200px;}
#MainMenu, footer{visibility:hidden;}
[data-testid="stSidebar"]{background:#0F1B2D;}
[data-testid="stSidebar"] *{color:#D5DEEC;}
[data-testid="stSidebar"] hr{border-color:rgba(255,255,255,.12);}
.brand{display:flex;align-items:center;gap:11px;margin:6px 0 20px 2px;}
.brand .mark{width:36px;height:36px;border-radius:9px;background:#0E8F80;display:flex;align-items:center;justify-content:center;font-weight:800;font-size:1.1rem;color:#fff;}
.brand .name{font-weight:800;font-size:1.2rem;color:#fff;line-height:1.1;}
.brand .sub{font-size:.78rem;color:#8FA0BA;}
[data-testid="stSidebar"] [role="radiogroup"]{gap:3px;}
[data-testid="stSidebar"] [role="radiogroup"] label{padding:9px 12px;border-radius:8px;width:100%;cursor:pointer;margin:0;}
[data-testid="stSidebar"] [role="radiogroup"] label:hover{background:rgba(255,255,255,.08);}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked){background:#0E8F80;}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) *{color:#fff !important;font-weight:700;}
[data-testid="stSidebar"] [role="radiogroup"] label[data-baseweb="radio"] > *:first-child:not(:has(p)){display:none;}
.side-note{font-size:.8rem;color:#8FA0BA !important;line-height:1.5;}
.hero{background:var(--surface);border:1px solid var(--line);border-left:5px solid var(--teal);border-radius:10px;padding:20px 26px;margin-bottom:22px;}
.hero-title{font-size:1.7rem;font-weight:800;letter-spacing:-.02em;color:var(--ink);line-height:1.2;}
.hero p{margin:6px 0 0 0;color:var(--muted);font-size:.98rem;}
.kpi{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:14px 16px;height:100%;}
.kpi .l{font-size:.85rem;color:var(--muted);font-weight:600;}
.kpi .v{font-size:1.8rem;font-weight:800;color:var(--ink);line-height:1.2;font-variant-numeric:tabular-nums;}
.kpi .s{font-size:.8rem;color:var(--muted);}
.feature{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:16px 18px;margin-bottom:14px;min-height:118px;}
.feature .t{font-weight:800;color:var(--ink);font-size:1.02rem;margin-bottom:4px;}
.feature .d{color:var(--muted);font-size:.9rem;line-height:1.5;}
.verdict{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:20px 22px;margin-bottom:10px;}
.verdict .tag{font-size:.85rem;color:var(--muted);font-weight:600;}
.verdict .name{font-size:2.05rem;font-weight:800;letter-spacing:-.02em;color:var(--ink);line-height:1.15;margin:2px 0 10px 0;text-transform:capitalize;word-break:break-word;}
.verdict .row{display:flex;align-items:baseline;gap:12px;margin-bottom:10px;}
.verdict .pct{font-size:2.5rem;font-weight:800;font-variant-numeric:tabular-nums;color:var(--c);line-height:1;}
.chip{display:inline-block;padding:3px 10px;border-radius:6px;font-size:.8rem;font-weight:700;color:var(--c);background:var(--bg);}
.meter{display:flex;gap:3px;height:12px;margin-bottom:12px;}
.meter i{flex:1;border-radius:2px;background:#E6EBF2;}
.meter i.on{background:var(--c);}
.verdict .meta{color:var(--muted);font-size:.85rem;}
.verdict .hint{margin-top:10px;padding:9px 12px;border-radius:8px;background:#FBF0DA;color:#7A4E05;font-size:.87rem;}
.empty{border:2px dashed #C3CEDB;border-radius:12px;padding:34px 20px;text-align:center;background:#fff;}
.empty .i{font-size:2rem;}
.empty .t{font-weight:800;color:var(--ink);margin-top:6px;}
.empty .d{color:var(--muted);font-size:.92rem;margin-top:2px;}
.chips span{display:inline-block;margin:0 6px 6px 0;padding:4px 11px;border-radius:6px;background:var(--teal-soft);color:#0A6D62;font-weight:700;font-size:.82rem;}
.stButton > button, .stDownloadButton > button{border-radius:8px;font-weight:700;}
[data-testid="stFileUploaderDropzone"], [data-testid="stFileUploadDropzone"]{border:2px dashed #9FB3C8;border-radius:12px;background:#fff;}
[data-testid="stTabs"] button[role="tab"]{font-weight:700;}
"""


def _minify(text: str) -> str:
    return "".join(line.strip() for line in text.strip().splitlines())


def md(markup: str) -> None:
    """Render an HTML snippet (newlines removed so Markdown never turns it into a code block)."""
    st.markdown(_minify(markup), unsafe_allow_html=True)


def esc(value) -> str:
    return html.escape(str(value))


def inject_css() -> None:
    st.markdown("<style>" + _minify(_CSS) + "</style>", unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Layout pieces
# --------------------------------------------------------------------------- #
def brand(name: str, tagline: str) -> None:
    md(f'<div class="brand"><div class="mark">V</div>'
       f'<div><div class="name">{esc(name)}</div><div class="sub">{esc(tagline)}</div></div></div>')


def hero(title: str, subtitle: str, icon: str = "") -> None:
    prefix = f"{icon} " if icon else ""
    md(f'<div class="hero"><div class="hero-title">{prefix}{esc(title)}</div><p>{esc(subtitle)}</p></div>')


def kpi_row(items) -> None:
    """items = [(label, value, optional_subtext), ...]"""
    cols = st.columns(len(items))
    for col, item in zip(cols, items):
        label, value = item[0], item[1]
        sub = item[2] if len(item) > 2 else ""
        sub_html = f'<div class="s">{esc(sub)}</div>' if sub else ""
        col.markdown(
            _minify(f'<div class="kpi"><div class="l">{esc(label)}</div>'
                    f'<div class="v">{esc(value)}</div>{sub_html}</div>'),
            unsafe_allow_html=True,
        )


def feature_card(title: str, text: str, icon: str = "") -> str:
    return _minify(f'<div class="feature"><div class="t">{icon} {esc(title)}</div>'
                   f'<div class="d">{esc(text)}</div></div>')


def empty_state(title: str, text: str, icon: str = "🗂️") -> None:
    md(f'<div class="empty"><div class="i">{icon}</div><div class="t">{esc(title)}</div>'
       f'<div class="d">{esc(text)}</div></div>')


def confidence_level(conf: float):
    """Return (name, text colour, soft background) for a confidence between 0 and 1."""
    if conf >= 0.8:
        return "High confidence", GREEN, "#E4F5EC"
    if conf >= 0.5:
        return "Medium confidence", AMBER, "#FBF0DA"
    return "Low confidence", CORAL, "#FBE6E8"


def verdict_html(label: str, conf: float, latency_ms: float | None = None, model: str | None = None) -> str:
    level, color, soft = confidence_level(conf)
    filled = round(conf * 20)
    segments = "".join('<i class="on"></i>' if i < filled else "<i></i>" for i in range(20))
    meta_parts = []
    if model:
        meta_parts.append(esc(model))
    if latency_ms is not None:
        meta_parts.append(f"{latency_ms:.0f} ms")
    meta = " · ".join(meta_parts)
    hint = ""
    if conf < 0.5:
        hint = ('<div class="hint">The model is unsure. This image may not match any of its '
                'categories, or it may be blurry or cropped.</div>')
    return _minify(
        f'<div class="verdict" style="--c:{color};--bg:{soft}">'
        f'<div class="tag">Predicted category</div>'
        f'<div class="name">{esc(label)}</div>'
        f'<div class="row"><span class="pct">{conf * 100:.1f}%</span>'
        f'<span class="chip">{level}</span></div>'
        f'<div class="meter">{segments}</div>'
        f'<div class="meta">{meta}</div>{hint}</div>'
    )


# --------------------------------------------------------------------------- #
# Charts (Plotly)
# --------------------------------------------------------------------------- #
def style_fig(fig: go.Figure, height: int = 320) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=height,
        margin=dict(l=8, r=8, t=36, b=8),
        font=dict(family="Manrope, Segoe UI, sans-serif", color=NAVY, size=13),
        colorway=PALETTE,
        legend=dict(orientation="h", y=-0.2, x=0),
    )
    fig.update_xaxes(gridcolor="#E6EBF2", zeroline=False)
    fig.update_yaxes(gridcolor="#E6EBF2", zeroline=False)
    return fig


def show_chart(fig: go.Figure, key: str, container=None) -> None:
    (container or st).plotly_chart(fig, theme=None, key=key)


def topk_chart(top_k) -> go.Figure:
    labels = [name for name, _ in top_k][::-1]
    values = [p * 100 for _, p in top_k][::-1]
    colors = [TEAL if i == len(values) - 1 else "#A9B8CC" for i in range(len(values))]
    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h", marker_color=colors,
        text=[f"{v:.1f}%" for v in values], textposition="outside", cliponaxis=False,
    ))
    fig.update_xaxes(range=[0, 112], title="Confidence (%)")
    return style_fig(fig, 70 + 36 * len(top_k))


def training_curves(history: list[dict]) -> go.Figure:
    epochs = [r["epoch"] for r in history]
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Loss", "Accuracy (%)"))
    styles = [("train_loss", "Train", TEAL, 1), ("val_loss", "Validation", CORAL, 1),
              ("train_acc", "Train", TEAL, 2), ("val_acc", "Validation", CORAL, 2)]
    for key, name, color, col in styles:
        values = [r[key] * 100 if col == 2 else r[key] for r in history]
        fig.add_trace(go.Scatter(x=epochs, y=values, name=name, mode="lines+markers",
                                 line=dict(color=color, width=3), legendgroup=name,
                                 showlegend=(col == 1)), row=1, col=col)
    fig.update_xaxes(title="Epoch", dtick=1)
    return style_fig(fig, 320)


# --------------------------------------------------------------------------- #
# Prediction display (used by Classify and Camera pages)
# --------------------------------------------------------------------------- #
def show_prediction(image, pred: dict, key: str, model: str | None = None, caption: str | None = None) -> None:
    col_img, col_res = st.columns([1, 1.4], gap="large")
    with col_img:
        st.image(image, caption=caption)
    with col_res:
        md(verdict_html(pred["label"], pred["confidence"], pred["latency_ms"], model))
        st.markdown("**Top predictions**")
        show_chart(topk_chart(pred["top_k"]), key=f"topk_{key}")
