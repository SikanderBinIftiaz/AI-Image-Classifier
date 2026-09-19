"""Quick end-to-end check that everything works on your machine. No downloads needed.

    python smoke_test.py

It builds a tiny colour dataset in a temp folder, trains a small model for a few
epochs, evaluates it, classifies an image and exercises the history database.
Your real data folders are not touched.
"""
import random
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src import config

# Send everything to a temp folder so your real data stays untouched.
TMP = Path(tempfile.mkdtemp(prefix="visionai_smoke_"))
config.DB_PATH = TMP / "history.db"
config.UPLOAD_DIR = TMP / "uploads"
config.MODELS_DIR = TMP / "models"
for folder in (config.UPLOAD_DIR, config.MODELS_DIR):
    folder.mkdir(parents=True, exist_ok=True)

from src import database, evaluation, models, training  # noqa: E402
from src.data import scan_dataset  # noqa: E402

COLORS = {"red": (200, 40, 40), "green": (40, 190, 60), "blue": (40, 70, 210)}


def make_image(color, seed):
    rng = np.random.default_rng(seed)
    base = np.array(color, dtype=np.float32)
    noise = rng.normal(0, 25, (64, 64, 3))
    return Image.fromarray(np.clip(base + noise, 0, 255).astype("uint8"))


def step(text):
    print(f"[ok] {text}")


def main():
    random.seed(0)
    # 1. Synthetic dataset -------------------------------------------------
    for split, n in (("train", 24), ("test", 8)):
        for name, color in COLORS.items():
            folder = TMP / "dataset" / split / name
            folder.mkdir(parents=True, exist_ok=True)
            for i in range(n):
                make_image(color, hash((split, name, i)) % 10_000).save(folder / f"{i}.png")
    root, counts = scan_dataset(TMP / "dataset")
    assert len(counts) == 3 and root.name == "train", counts
    step(f"dataset scanned: {counts}")

    # 2. Every architecture builds and runs ---------------------------------
    for arch in models.MODEL_ZOO:
        net = models.build_backbone(arch, num_classes=5).eval()
        with torch.no_grad():
            assert net(torch.zeros(1, 3, 64, 64)).shape == (1, 5)
    step("all architectures build with a new final layer")

    # 3. Training -----------------------------------------------------------
    events = []
    meta = training.train_custom_model(
        TMP / "dataset", "smoke", arch="ResNet18", epochs=4, batch_size=8, lr=3e-3,
        val_split=0.25, img_size=64, freeze_backbone=False, pretrained=False,
        progress_cb=events.append)
    assert (config.MODELS_DIR / "smoke" / "model.pt").exists()
    assert any(e["type"] == "epoch" for e in events)
    step(f"trained custom model, best validation accuracy {meta['best_val_acc'] * 100:.0f}%")

    # 4. Load + predict -----------------------------------------------------
    name = models.CUSTOM_PREFIX + "smoke"
    assert name in models.available_models()
    loaded = models.load_model(name)
    pred = models.predict(loaded, make_image(COLORS["red"], 1), top_k=3)
    assert set(pred) == {"label", "confidence", "top_k", "latency_ms"} and len(pred["top_k"]) == 3
    step(f"prediction works: {pred['label']} ({pred['confidence'] * 100:.0f}%)")

    # 5. Evaluation ---------------------------------------------------------
    result = evaluation.evaluate_folder(loaded, TMP / "dataset" / "test")
    assert result["n_images"] == 24 and result["confusion"].shape == (3, 3)
    step(f"evaluation works: accuracy {result['accuracy'] * 100:.0f}%, F1 {result['f1'] * 100:.0f}%")

    # 6. History database ---------------------------------------------------
    database.init_db()
    row_id = database.add_prediction(make_image(COLORS["red"], 2), "test.png", name, pred)
    history = database.fetch_history()
    assert len(history) == 1 and history.loc[0, "label"] == pred["label"]
    database.delete_prediction(row_id)
    assert database.fetch_history().empty
    step("history database works")

    print("\nALL CHECKS PASSED. Run the app with:  streamlit run app.py")


if __name__ == "__main__":
    main()
