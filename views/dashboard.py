"""Results dashboard: charts built from the saved prediction history."""
import plotly.express as px
import streamlit as st

from src import database, ui


def render() -> None:
    ui.hero("Results dashboard", "A summary of everything you have classified so far.", "📊")

    df = database.fetch_history()
    if df.empty:
        ui.empty_state("No data yet", "Classify a few images and your charts will appear here.", "📊")
        return

    high = (df["confidence_pct"] >= 80).mean() * 100
    ui.kpi_row([
        ("Total predictions", f"{len(df):,}"),
        ("Average confidence", f"{df['confidence_pct'].mean():.1f}%", f"{high:.0f}% are above 80%"),
        ("Average speed", f"{df['latency_ms'].mean():.0f} ms", "per image"),
        ("Different categories", f"{df['label'].nunique()}"),
    ])
    st.write("")

    left, right = st.columns(2)
    with left:
        st.markdown("**Predictions per day**")
        daily = (df.assign(day=df["created_at"].dt.strftime("%Y-%m-%d"))
                   .groupby("day").size().reset_index(name="predictions"))
        fig = px.bar(daily, x="day", y="predictions", color_discrete_sequence=[ui.TEAL])
        fig.update_xaxes(title="", type="category")
        fig.update_yaxes(title="", dtick=1 if daily["predictions"].max() < 8 else None)
        ui.show_chart(ui.style_fig(fig, 300), "dash_daily")
    with right:
        st.markdown("**Most frequent categories**")
        top = df["label"].value_counts().head(10).rename_axis("label").reset_index(name="count")
        fig = px.bar(top, x="count", y="label", orientation="h", color_discrete_sequence=[ui.NAVY])
        fig.update_yaxes(title="", autorange="reversed")
        fig.update_xaxes(title="")
        ui.show_chart(ui.style_fig(fig, 300), "dash_top")

    left, right = st.columns(2)
    with left:
        st.markdown("**How confident is the model?**")
        fig = px.histogram(df, x="confidence_pct", nbins=20, range_x=[0, 100],
                           color_discrete_sequence=[ui.TEAL])
        fig.update_xaxes(title="Confidence (%)")
        fig.update_yaxes(title="Images")
        ui.show_chart(ui.style_fig(fig, 300), "dash_conf")
    with right:
        st.markdown("**Model usage**")
        usage = df["model"].value_counts().rename_axis("model").reset_index(name="count")
        fig = px.pie(usage, names="model", values="count", hole=0.6, color_discrete_sequence=ui.PALETTE)
        ui.show_chart(ui.style_fig(fig, 300), "dash_usage")

    st.markdown("**Confidence and speed by model**")
    per_model = (df.groupby("model")
                   .agg(images=("id", "count"), avg_confidence=("confidence_pct", "mean"),
                        avg_latency_ms=("latency_ms", "mean"))
                   .round(1).reset_index())
    st.dataframe(per_model, hide_index=True)

    st.markdown("**Latest predictions**")
    latest = df.head(10)[["created_at", "filename", "label", "confidence_pct", "model"]]
    st.dataframe(
        latest, hide_index=True,
        column_config={
            "created_at": st.column_config.DatetimeColumn("When", format="DD MMM, HH:mm"),
            "filename": "File", "label": "Prediction", "model": "Model",
            "confidence_pct": st.column_config.ProgressColumn("Confidence", min_value=0, max_value=100, format="%.1f%%"),
        },
    )
