"""Model evaluation page: metrics, confusion matrix and per-class results."""
import hashlib
import io

import plotly.express as px
import streamlit as st

from src import config, data, evaluation, models, ui


def _test_folder() -> str | None:
    """Let the user choose a test folder or upload a ZIP. Returns a folder path or None."""
    source = st.radio("Test images", ["Folder on this computer", "Upload a ZIP file"], horizontal=True)
    if source == "Upload a ZIP file":
        zip_file = st.file_uploader("ZIP with one sub-folder per class", type=["zip"], key="eval_zip")
        if zip_file is None:
            return None
        digest = hashlib.md5(zip_file.getvalue()).hexdigest()[:8]
        dest = config.DATASET_DIR / f"eval_{digest}"
        if not dest.exists():
            try:
                data.extract_zip(io.BytesIO(zip_file.getvalue()), dest)
            except Exception as exc:
                st.error(f"Could not read the ZIP file: {exc}")
                return None
        return str(dest)

    default = config.DATA_DIR / "sample_dataset" / "test"
    return st.text_input("Path to the test folder", value=str(default) if default.exists() else "",
                         placeholder=r"C:\datasets\my_test_set   or   /home/me/my_test_set",
                         help="The folder must contain one sub-folder per class, each holding images.") or None


def _show_results(res: dict) -> None:
    ui.kpi_row([
        ("Accuracy", f"{res['accuracy'] * 100:.1f}%", f"{res['n_images']:,} images tested"),
        ("Precision", f"{res['precision'] * 100:.1f}%", "weighted average"),
        ("Recall", f"{res['recall'] * 100:.1f}%", "weighted average"),
        ("F1 score", f"{res['f1'] * 100:.1f}%", f"Top-5 accuracy {res['top5'] * 100:.1f}%"),
    ])
    if res["skipped"]:
        st.warning("Skipped folders that do not match any model class: " + ", ".join(res["skipped"][:10])
                   + (" ..." if len(res["skipped"]) > 10 else ""))
    st.write("")

    tab_cm, tab_cls, tab_train = st.tabs(["Confusion matrix", "Per-class results", "Training curves"])
    with tab_cm:
        normalise = st.toggle("Show percentages per true class", value=True)
        cm = res["confusion"].astype(float)
        if normalise:
            cm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
        labels = res["labels"]
        fig = px.imshow(cm, x=labels, y=labels, color_continuous_scale=[[0, "#F4F6F9"], [1, ui.TEAL]],
                        text_auto=".0%" if normalise else True, aspect="auto",
                        labels=dict(x="Predicted", y="True", color=""))
        fig.update_xaxes(side="bottom", showgrid=False)
        fig.update_yaxes(showgrid=False)
        fig.update_coloraxes(showscale=False)
        ui.show_chart(ui.style_fig(fig, max(380, 34 * len(labels))), "eval_cm")
        st.caption("Rows are the true class and columns are what the model predicted. "
                   "A strong diagonal means the model is getting most images right.")
    with tab_cls:
        table = res["per_class"].copy()
        for col in ("precision", "recall", "f1"):
            table[col] = (table[col] * 100).round(1)
        st.dataframe(table, hide_index=True, column_config={
            "class": "Class",
            "precision": st.column_config.ProgressColumn("Precision", min_value=0, max_value=100, format="%.1f%%"),
            "recall": st.column_config.ProgressColumn("Recall", min_value=0, max_value=100, format="%.1f%%"),
            "f1": st.column_config.ProgressColumn("F1", min_value=0, max_value=100, format="%.1f%%"),
            "support": "Images",
        })
        st.download_button("Download report (CSV)", table.to_csv(index=False).encode("utf-8"),
                           file_name="evaluation_report.csv", mime="text/csv")
    with tab_train:
        if res["model"].startswith(models.CUSTOM_PREFIX):
            meta = models.read_meta(res["model"][len(models.CUSTOM_PREFIX):])
            ui.show_chart(ui.training_curves(meta["history"]), "eval_curves")
            st.caption(f"{meta['arch']} trained for {meta['epochs']} epochs on {meta['train_images']:,} images. "
                       f"Best validation accuracy: {meta['best_val_acc'] * 100:.1f}%.")
        else:
            st.info("Training curves are available for models you train yourself on the Train page.")


def render() -> None:
    ui.hero("Model evaluation",
            "Measure how well a model performs on images it has never seen.", "🧪")

    c1, c2 = st.columns([1.3, 1])
    model_name = c1.selectbox("Model to evaluate", models.available_models())
    max_images = c2.number_input("Maximum images to test (0 = all)", min_value=0, value=500, step=100)
    folder = _test_folder()

    if st.button("Run evaluation", type="primary", disabled=not folder):
        bar = st.progress(0.0, text="Preparing...")
        try:
            loaded = models.load_model(model_name)
            result = evaluation.evaluate_folder(
                loaded, folder, max_images=int(max_images),
                progress_cb=lambda f: bar.progress(min(f, 1.0), text=f"Testing images... {f * 100:.0f}%"))
            st.session_state["eval_result"] = result
        except Exception as exc:
            st.session_state.pop("eval_result", None)
            st.error(str(exc))
        finally:
            bar.empty()

    result = st.session_state.get("eval_result")
    if result:
        st.subheader(f"Results for {result['model']}")
        _show_results(result)
    else:
        ui.empty_state("No evaluation yet",
                       "Choose a model and a test folder, then press Run evaluation.", "🧪")
