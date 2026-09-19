"""Train a CNN on your own dataset with transfer learning (PyTorch)."""
from __future__ import annotations

import json
import random
import re
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from . import config
from .data import FolderDataset, scan_dataset
from .models import DEVICE, build_backbone, eval_transform, train_transform


def safe_name(name: str) -> str:
    """Turn a model name into something safe to use as a folder name."""
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", name.strip()).strip("_")
    return cleaned or "my_model"


def _run_epoch(model, loader, criterion, optimizer=None, freeze_bn=False, on_batch=None):
    """One pass over `loader`. Trains when an optimizer is given, otherwise evaluates."""
    training = optimizer is not None
    model.train(training)
    if training and freeze_bn:  # keep pretrained BatchNorm statistics untouched
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()

    total_loss, correct, seen = 0.0, 0, 0
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for i, (x, y) in enumerate(loader, 1):
            x, y = x.to(DEVICE), y.to(DEVICE)
            if training:
                optimizer.zero_grad(set_to_none=True)
            out = model(x)
            loss = criterion(out, y)
            if training:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * x.size(0)
            correct += (out.argmax(dim=1) == y).sum().item()
            seen += x.size(0)
            if on_batch:
                on_batch(i / len(loader))
    return total_loss / seen, correct / seen


def train_custom_model(data_dir, model_name: str, arch: str = "MobileNetV2", epochs: int = 5,
                       batch_size: int = 32, lr: float = 1e-3, val_split: float = 0.2,
                       img_size: int = 224, freeze_backbone: bool = True,
                       pretrained: bool = True, progress_cb=None, seed: int = 42) -> dict:
    """Train on a folder with one sub-folder per class and save the best model.

    progress_cb receives dicts:
      {"type": "batch", "epoch": e, "epochs": n, "phase": "train"|"val", "fraction": 0..1}
      {"type": "epoch", "epoch": e, "epochs": n, "row": {...metrics...}}
    """
    torch.manual_seed(seed)
    root, counts = scan_dataset(data_dir)
    classes = sorted(counts)
    if len(classes) < 2:
        raise ValueError("A dataset needs at least 2 class folders that contain images.")

    full = FolderDataset(root, classes)
    indices = list(range(len(full)))
    random.Random(seed).shuffle(indices)
    n_val = max(1, int(len(indices) * val_split))
    val_idx, train_idx = indices[:n_val], indices[n_val:]
    if not train_idx:
        raise ValueError("Not enough images: lower the validation split or add more images.")

    train_ds = Subset(full.with_transform(train_transform(img_size)), train_idx)
    val_ds = Subset(full.with_transform(eval_transform(img_size)), val_idx)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0,
                              drop_last=len(train_ds) > batch_size)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = build_backbone(arch, len(classes), pretrained=pretrained,
                           freeze_backbone=freeze_backbone).to(DEVICE)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    criterion = nn.CrossEntropyLoss()

    history, best_acc, best_state = [], -1.0, None
    for epoch in range(1, epochs + 1):
        def batch_cb(phase):
            def _cb(fraction):
                if progress_cb:
                    progress_cb({"type": "batch", "epoch": epoch, "epochs": epochs,
                                 "phase": phase, "fraction": fraction})
            return _cb

        tr_loss, tr_acc = _run_epoch(model, train_loader, criterion, optimizer,
                                     freeze_bn=freeze_backbone, on_batch=batch_cb("train"))
        va_loss, va_acc = _run_epoch(model, val_loader, criterion, on_batch=batch_cb("val"))
        scheduler.step()

        row = {"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc,
               "val_loss": va_loss, "val_acc": va_acc}
        history.append(row)
        if va_acc > best_acc:
            best_acc = va_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if progress_cb:
            progress_cb({"type": "epoch", "epoch": epoch, "epochs": epochs, "row": row})

    folder_name = safe_name(model_name)
    out_dir = config.MODELS_DIR / folder_name
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, out_dir / "model.pt")
    meta = {
        "name": folder_name,
        "arch": arch,
        "classes": classes,
        "img_size": img_size,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "freeze_backbone": freeze_backbone,
        "train_images": len(train_idx),
        "val_images": len(val_idx),
        "best_val_acc": best_acc,
        "history": history,
        "created": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta
