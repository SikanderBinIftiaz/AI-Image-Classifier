"""Model zoo, custom-model loading and prediction (PyTorch / torchvision).

Nothing in this file depends on Streamlit, so the same code powers the web app,
the live-camera script and the smoke test.
"""
from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models as tv
from torchvision import transforms as T

from . import config
from .data import load_image  # noqa: F401  (re-exported for convenience)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# name -> (torchvision builder, pretrained ImageNet weights)
MODEL_ZOO = {
    "MobileNetV2": (tv.mobilenet_v2, tv.MobileNet_V2_Weights.IMAGENET1K_V1),
    "ResNet18": (tv.resnet18, tv.ResNet18_Weights.IMAGENET1K_V1),
    "ResNet50": (tv.resnet50, tv.ResNet50_Weights.IMAGENET1K_V2),
    "EfficientNet-B0": (tv.efficientnet_b0, tv.EfficientNet_B0_Weights.IMAGENET1K_V1),
}
CUSTOM_PREFIX = "Custom: "


# --------------------------------------------------------------------------- #
# Transforms
# --------------------------------------------------------------------------- #
def eval_transform(img_size: int = 224) -> T.Compose:
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def train_transform(img_size: int = 224) -> T.Compose:
    """Augmentation used while training on a custom dataset."""
    return T.Compose([
        T.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


# --------------------------------------------------------------------------- #
# Building networks
# --------------------------------------------------------------------------- #
def build_backbone(arch: str, num_classes: int | None = None, pretrained: bool = False,
                   freeze_backbone: bool = False) -> nn.Module:
    """Create a CNN. With `num_classes`, the last layer is replaced (transfer learning)."""
    builder, weights = MODEL_ZOO[arch]
    model = builder(weights=weights if pretrained else None)
    if freeze_backbone:
        for p in model.parameters():
            p.requires_grad = False
    if num_classes is not None:
        if hasattr(model, "fc"):                       # ResNet family
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        else:                                          # MobileNet / EfficientNet
            model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    return model


@dataclass
class LoadedModel:
    name: str
    model: nn.Module
    classes: list[str]
    preprocess: T.Compose
    custom: bool = False
    params_m: float = 0.0
    info: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Custom models (trained by the user)
# --------------------------------------------------------------------------- #
def list_custom_models() -> list[str]:
    found = []
    for folder in sorted(config.MODELS_DIR.iterdir()):
        if (folder / "meta.json").exists() and (folder / "model.pt").exists():
            found.append(folder.name)
    return found


def available_models() -> list[str]:
    return list(MODEL_ZOO) + [CUSTOM_PREFIX + n for n in list_custom_models()]


def read_meta(custom_name: str) -> dict:
    return json.loads((config.MODELS_DIR / custom_name / "meta.json").read_text(encoding="utf-8"))


def delete_custom_model(custom_name: str) -> None:
    shutil.rmtree(config.MODELS_DIR / custom_name, ignore_errors=True)


# --------------------------------------------------------------------------- #
# Loading (with an in-memory cache so a model is only built once)
# --------------------------------------------------------------------------- #
_CACHE: dict[tuple, LoadedModel] = {}


def _cache_key(name: str) -> tuple:
    if name.startswith(CUSTOM_PREFIX):
        weights = config.MODELS_DIR / name[len(CUSTOM_PREFIX):] / "model.pt"
        return (name, weights.stat().st_mtime_ns if weights.exists() else 0)
    return (name, 0)


def load_model(name: str) -> LoadedModel:
    key = _cache_key(name)
    if key in _CACHE:
        return _CACHE[key]
    for stale in [k for k in _CACHE if k[0] == name]:
        del _CACHE[stale]  # a retrained custom model replaces the old one

    info: dict = {}
    if name in MODEL_ZOO:
        builder, weights = MODEL_ZOO[name]
        model = builder(weights=weights)          # downloads the weights the first time
        classes = list(weights.meta["categories"])
        preprocess = weights.transforms()
        custom = False
        acc = weights.meta.get("_metrics", {}).get("ImageNet-1K", {}).get("acc@1")
        info = {"imagenet_top1": acc}
    elif name.startswith(CUSTOM_PREFIX):
        folder_name = name[len(CUSTOM_PREFIX):]
        meta = read_meta(folder_name)
        model = build_backbone(meta["arch"], len(meta["classes"]), pretrained=False)
        state = torch.load(config.MODELS_DIR / folder_name / "model.pt", map_location="cpu",
                           weights_only=True)
        model.load_state_dict(state)
        classes = list(meta["classes"])
        preprocess = eval_transform(meta.get("img_size", 224))
        custom = True
        info = {"arch": meta["arch"], "best_val_acc": meta.get("best_val_acc")}
    else:
        raise ValueError(f"Unknown model: {name}")

    model.to(DEVICE).eval()
    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    loaded = LoadedModel(name, model, classes, preprocess, custom, params_m, info)
    _CACHE[key] = loaded
    return loaded


# --------------------------------------------------------------------------- #
# Prediction
# --------------------------------------------------------------------------- #
@torch.inference_mode()
def predict_probs(loaded: LoadedModel, image: Image.Image) -> tuple[np.ndarray, float]:
    """Return (probability for every class, inference time in milliseconds)."""
    x = loaded.preprocess(image.convert("RGB")).unsqueeze(0).to(DEVICE)
    start = time.perf_counter()
    logits = loaded.model(x)
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    latency_ms = (time.perf_counter() - start) * 1000
    probs = torch.softmax(logits.float(), dim=1)[0].cpu().numpy()
    return probs, latency_ms


def top_k_from_probs(probs: np.ndarray, classes: list[str], k: int = 5) -> list[tuple[str, float]]:
    k = max(1, min(k, len(probs)))
    order = np.argsort(probs)[::-1][:k]
    return [(classes[i], float(probs[i])) for i in order]


def predict(loaded: LoadedModel, image: Image.Image, top_k: int = 5, warmup: bool = False) -> dict:
    """Classify one image. `warmup=True` runs once first so the latency is realistic."""
    if warmup:
        predict_probs(loaded, image)
    probs, latency_ms = predict_probs(loaded, image)
    top = top_k_from_probs(probs, loaded.classes, top_k)
    return {
        "label": top[0][0],
        "confidence": top[0][1],
        "top_k": top,
        "latency_ms": latency_ms,
    }
