"""Create a small ready-to-use dataset so you can try training and evaluation right away.

Downloads CIFAR-10 (about 170 MB, once) and saves a subset as image files:

    data/sample_dataset/train/<category>/*.png
    data/sample_dataset/test/<category>/*.png

Usage:  python make_sample_dataset.py
        python make_sample_dataset.py --train-per-class 400 --test-per-class 100
"""
import argparse
from pathlib import Path

from PIL import Image
from torchvision import datasets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-per-class", type=int, default=200)
    parser.add_argument("--test-per-class", type=int, default=50)
    parser.add_argument("--size", type=int, default=96, help="saved image size in pixels")
    args = parser.parse_args()

    base = Path(__file__).resolve().parent / "data"
    out_dir = base / "sample_dataset"
    download_dir = base / "_cifar10_download"

    for split, per_class in (("train", args.train_per_class), ("test", args.test_per_class)):
        dataset = datasets.CIFAR10(root=str(download_dir), train=(split == "train"), download=True)
        saved = {name: 0 for name in dataset.classes}
        for i in range(len(dataset)):
            image, label = dataset[i]
            name = dataset.classes[label]
            if saved[name] >= per_class:
                continue
            folder = out_dir / split / name
            folder.mkdir(parents=True, exist_ok=True)
            image.resize((args.size, args.size), Image.BICUBIC).save(folder / f"{name}_{saved[name]:04d}.png")
            saved[name] += 1
            if all(v >= per_class for v in saved.values()):
                break
        print(f"{split}: {sum(saved.values())} images in {len(saved)} categories -> {out_dir / split}")

    print("\nDone. In the app: Train -> 'Sample dataset', then Evaluate -> the test folder.")


if __name__ == "__main__":
    main()
