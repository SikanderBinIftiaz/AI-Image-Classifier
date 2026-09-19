"""Image + dataset helpers. No Streamlit code in here, so it is easy to test."""
from __future__ import annotations

import copy
import zipfile
from pathlib import Path

from PIL import Image, ImageOps
from torch.utils.data import Dataset

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
_SKIP_PREFIX = (".", "__")  # hidden folders and __MACOSX


def load_image(src) -> Image.Image:
    """Open an image from a path or file-like object as an upright RGB image."""
    img = Image.open(src)
    img = ImageOps.exif_transpose(img)  # fixes phone photos that are rotated
    return img.convert("RGB")


def _is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMG_EXT


def _class_dirs(root: Path) -> list[Path]:
    return sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith(_SKIP_PREFIX))


def resolve_dataset_root(path) -> Path:
    """Return the folder that directly contains one sub-folder per class.

    Handles the two common surprises: a ZIP that wraps everything in one extra
    folder, and a dataset that is split into train/ and test/ folders.
    """
    root = Path(str(path).strip().strip('"')).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"Folder not found: {root}")
    for _ in range(4):
        dirs = _class_dirs(root)
        by_name = {d.name.lower(): d for d in dirs}
        train = by_name.get("train") or by_name.get("training")
        if train is not None:
            root = train
            continue
        has_images = any(_is_image(p) for p in root.iterdir())
        if len(dirs) == 1 and not has_images and any(c.is_dir() for c in dirs[0].iterdir()):
            root = dirs[0]
            continue
        break
    return root


def scan_dataset(path) -> tuple[Path, dict[str, int]]:
    """Return (dataset_root, {class_name: image_count}) for a class-per-folder dataset."""
    root = resolve_dataset_root(path)
    counts: dict[str, int] = {}
    for d in _class_dirs(root):
        n = sum(1 for f in d.rglob("*") if _is_image(f))
        if n:
            counts[d.name] = n
    return root, counts


class FolderDataset(Dataset):
    """PyTorch dataset for  root/<class_name>/<image files>."""

    def __init__(self, root, classes=None, transform=None):
        self.root = Path(root)
        if classes is None:
            _, counts = scan_dataset(self.root)
            classes = sorted(counts)
        self.classes = list(classes)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.samples: list[tuple[Path, int]] = []
        for c in self.classes:
            for f in sorted((self.root / c).rglob("*")):
                if _is_image(f):
                    self.samples.append((f, self.class_to_idx[c]))
        if not self.samples:
            raise ValueError(f"No images found in {self.root}")
        self.targets = [t for _, t in self.samples]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i):
        path, target = self.samples[i]
        img = load_image(path)
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def with_transform(self, transform) -> "FolderDataset":
        clone = copy.copy(self)  # shares the file list, different transform
        clone.transform = transform
        return clone


def extract_zip(file_obj, dest: Path) -> Path:
    """Safely extract an uploaded ZIP into `dest` (blocks path-traversal tricks)."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    base = dest.resolve()
    with zipfile.ZipFile(file_obj) as zf:
        for member in zf.infolist():
            if not (base / member.filename).resolve().is_relative_to(base):
                raise ValueError("The ZIP file contains unsafe paths.")
        zf.extractall(dest)
    return dest
