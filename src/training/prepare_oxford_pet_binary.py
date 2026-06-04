"""Prepare a local Oxford-IIIT Pet cats-vs-dogs training folder."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from src.training.metrics import count_labels, sha256_file

SOURCE_URL = "https://www.robots.ox.ac.uk/~vgg/data/pets/"
LICENSE_URL = "https://creativecommons.org/licenses/by-sa/4.0/"
TORCHVISION_DOC_URL = "https://docs.pytorch.org/vision/main/generated/torchvision.datasets.OxfordIIITPet.html"


def prepare_dataset(
    root: str | Path,
    output_dir: str | Path,
    split: str = "trainval",
    max_samples_per_class: int | None = None,
    copy_images: bool = True,
) -> dict[str, object]:
    try:
        from torchvision.datasets import OxfordIIITPet
    except ImportError as exc:
        raise RuntimeError("torchvision is required to prepare Oxford-IIIT Pet data") from exc

    root_path = Path(root)
    output_path = Path(output_dir)
    images_dir = output_path / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    dataset = OxfordIIITPet(
        root=str(root_path),
        split=split,
        target_types="binary-category",
        download=True,
    )

    rows: list[tuple[str, str, str]] = []
    counts = {"cat": 0, "dog": 0}

    for index in range(len(dataset)):
        image, target = dataset[index]
        label = "cat" if int(target) == 0 else "dog"
        if max_samples_per_class is not None and counts[label] >= max_samples_per_class:
            continue

        filename = f"{split}_{index:05d}_{label}.jpg"
        target_path = images_dir / filename
        if copy_images:
            image.save(target_path, format="JPEG", quality=95)
        else:
            image.save(target_path, format="JPEG", quality=95)

        rows.append((filename, label, sha256_file(target_path)))
        counts[label] += 1

    labels_path = output_path / "labels.csv"
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    labels_path.write_text(
        "image_name,label,sha256\n"
        + "\n".join(f"{filename},{label},{checksum}" for filename, label, checksum in rows)
        + "\n",
        encoding="utf-8",
    )

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": "Oxford-IIIT Pet Dataset via torchvision.datasets.OxfordIIITPet",
        "source_url": SOURCE_URL,
        "torchvision_docs": TORCHVISION_DOC_URL,
        "license": "Creative Commons Attribution-ShareAlike 4.0 International",
        "license_url": LICENSE_URL,
        "split": split,
        "target_types": "binary-category",
        "output_dir": str(output_path),
        "labels_csv": str(labels_path),
        "samples": len(rows),
        "class_counts": count_labels(label for _, label, _ in rows),
        "max_samples_per_class": max_samples_per_class,
        "data_policy": "Prepared data stays in ignored data/ or outputs/ folders and is not committed to git.",
    }
    manifest_path = output_path / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Oxford-IIIT Pet as a binary cats-vs-dogs dataset")
    parser.add_argument("--root", default="data/raw", help="Download/cache root used by torchvision")
    parser.add_argument("--output-dir", default="data/oxford_pet_binary", help="Prepared labels/images output folder")
    parser.add_argument("--split", default="trainval", choices=["trainval", "test"])
    parser.add_argument("--max-samples-per-class", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    manifest = prepare_dataset(
        root=args.root,
        output_dir=args.output_dir,
        split=args.split,
        max_samples_per_class=args.max_samples_per_class,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
