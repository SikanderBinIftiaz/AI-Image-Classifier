"""History page: browse, filter, export and delete saved predictions."""
from pathlib import Path

import streamlit as st

from src import database, ui


def render() -> None:
    ui.hero("Prediction history", "Every saved result, with its image. Filter, export or delete.", "🕘")

    df = database.fetch_history()
    if df.empty:
        ui.empty_state("History is empty", "Results are saved automatically when you classify images.", "🕘")
        return

    f1, f2, f3 = st.columns([1.4, 1.4, 1])
    models_selected = f1.multiselect("Model", sorted(df["model"].unique()))
    search = f2.text_input("Search category or file name", placeholder="e.g. golden retriever")
    min_conf = f3.slider("Minimum confidence (%)", 0, 100, 0)

    view = df
    if models_selected:
        view = view[view["model"].isin(models_selected)]
    if search.strip():
        q = search.strip()
        view = view[view["label"].str.contains(q, case=False, na=False)
                    | view["filename"].str.contains(q, case=False, na=False)]
    view = view[view["confidence_pct"] >= min_conf]

    st.caption(f"Showing {len(view):,} of {len(df):,} saved predictions")
    if view.empty:
        ui.empty_state("No matches", "Try removing a filter.", "🔍")
    else:
        gallery, table = st.tabs(["Gallery", "Table"])
        with gallery:
            limit = st.slider("Images to show", 8, 96, 24, step=4)
            records = list(view.head(limit).itertuples())
            for start in range(0, len(records), 4):
                cols = st.columns(4)
                for col, row in zip(cols, records[start:start + 4]):
                    with col.container(border=True):
                        if row.image_path and Path(row.image_path).exists():
                            st.image(row.image_path)
                        st.markdown(f"**{row.label}**")
                        st.caption(f"{row.confidence_pct:.1f}% · {row.model}  \n{row.created_at:%d %b %Y, %H:%M}")
                        if st.button("Delete", key=f"del_{row.id}"):
                            database.delete_prediction(row.id)
                            st.rerun()
        with table:
            shown = view[["created_at", "filename", "model", "label", "confidence_pct", "latency_ms", "source"]]
            st.dataframe(
                shown, hide_index=True,
                column_config={
                    "created_at": st.column_config.DatetimeColumn("When", format="DD MMM YYYY, HH:mm"),
                    "filename": "File", "model": "Model", "label": "Prediction", "source": "Source",
                    "latency_ms": st.column_config.NumberColumn("Speed (ms)", format="%.0f"),
                    "confidence_pct": st.column_config.ProgressColumn("Confidence", min_value=0, max_value=100, format="%.1f%%"),
                },
            )
            st.download_button("Download as CSV", shown.to_csv(index=False).encode("utf-8"),
                               file_name="prediction_history.csv", mime="text/csv")

    with st.expander("Delete all history"):
        confirm = st.checkbox("Yes, permanently delete every saved prediction")
        if st.button("Delete everything", disabled=not confirm):
            database.clear_history()
            st.rerun()
