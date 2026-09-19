"""Evaluate a model on a labelled test folder (one sub-folder per class)."""
from __future__ import annotations

import random

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader, Subset

from .data import FolderDataset, scan_dataset
from .models import DEVICE, LoadedModel


def _norm(text: str) -> str:
    return text.lower().replace("_", " ").replace("-", " ").strip()


def build_label_map(folder_classes: list[str], model_classes: list[str]) -> dict[int, int]:
    """Match test-folder names to the model's class names (ignores case, _ and -)."""
    index: dict[str, int] = {}
    for i, name in enumerate(model_classes):
        index.setdefault(_norm(name), i)
    mapping = {}
    for i, name in enumerate(folder_classes):
        j = index.get(_norm(name))
        if j is not None:
            mapping[i] = j
    return mapping


def evaluate_folder(loaded: LoadedModel, folder, batch_size: int = 32, max_images: int = 0,
                    progress_cb=None, seed: int = 42) -> dict:
    root, counts = scan_dataset(folder)
    if not counts:
        raise ValueError("No class sub-folders containing images were found in that folder.")

    dataset = FolderDataset(root, sorted(counts), transform=loaded.preprocess)
    mapping = build_label_map(dataset.classes, loaded.classes)
    if not mapping:
        raise ValueError(
            "None of the folder names match this model's classes. Custom models need test folders "
            "named exactly like the training classes; ImageNet models need ImageNet class names "
            "(for example 'goldfish' or 'tabby')."
        )
    skipped = [c for i, c in enumerate(dataset.classes) if i not in mapping]

    keep = [i for i, target in enumerate(dataset.targets) if target in mapping]
    if max_images and len(keep) > max_images:
        keep = sorted(random.Random(seed).sample(keep, max_images))
    loader = DataLoader(Subset(dataset, keep), batch_size=batch_size, shuffle=False, num_workers=0)

    y_true: list[int] = []
    y_pred: list[int] = []
    top5_hits = 0
    loaded.model.eval()
    with torch.inference_mode():
        for step, (x, target) in enumerate(loader, 1):
            logits = loaded.model(x.to(DEVICE))
            k = min(5, logits.shape[1])
            top = logits.topk(k, dim=1).indices.cpu()
            truth = torch.tensor([mapping[int(t)] for t in target])
            y_true += truth.tolist()
            y_pred += top[:, 0].tolist()
            top5_hits += (top == truth.unsqueeze(1)).any(dim=1).sum().item()
            if progress_cb:
                progress_cb(step / len(loader))

    truth_arr, pred_arr = np.array(y_true), np.array(y_pred)
    labels = sorted(set(y_true))
    names = [loaded.classes[i] for i in labels]
    cm = confusion_matrix(truth_arr, pred_arr, labels=labels)
    prec, rec, f1, support = precision_recall_fscore_support(
        truth_arr, pred_arr, labels=labels, zero_division=0)
    w_prec, w_rec, w_f1, _ = precision_recall_fscore_support(
        truth_arr, pred_arr, labels=labels, average="weighted", zero_division=0)

    per_class = pd.DataFrame({
        "class": names, "precision": prec, "recall": rec, "f1": f1, "support": support,
    })
    return {
        "model": loaded.name,
        "n_images": len(y_true),
        "accuracy": float((truth_arr == pred_arr).mean()),
        "top5": top5_hits / len(y_true),
        "precision": float(w_prec),
        "recall": float(w_rec),
        "f1": float(w_f1),
        "labels": names,
        "confusion": cm,
        "per_class": per_class,
        "skipped": skipped,
    }
