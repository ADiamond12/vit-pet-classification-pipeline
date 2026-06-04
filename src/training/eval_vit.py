"""Evaluate a local ViT checkpoint and write public-review artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import ViTForImageClassification, ViTImageProcessor

from src.training.metrics import (
    LABELS,
    compute_classification_metrics,
    utc_now_iso,
    write_evaluation_artifacts,
)
from src.training.train_vit import DEFAULT_FILENAME_COL, DEFAULT_LABEL_COL, LABEL2ID, filter_existing_images


class CatDogEvalDataset(Dataset):
    def __init__(self, csv_path: str | Path, images_dir: str | Path, filename_col: str, label_col: str) -> None:
        df = pd.read_csv(csv_path)
        df = filter_existing_images(df, images_dir, filename_col)
        df[label_col] = df[label_col].astype(str).str.strip().str.lower()
        unknown = sorted(set(df[label_col]) - set(LABEL2ID))
        if unknown:
            raise ValueError(f"Unsupported labels: {unknown}")

        self.data = df.reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.filename_col = filename_col
        self.label_col = label_col

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.data.iloc[index]
        image = Image.open(self.images_dir / str(row[self.filename_col])).convert("RGB")
        return {"image": image, "label": LABEL2ID[str(row[self.label_col]).strip().lower()]}


def validation_subset(dataset: Dataset, val_split: float, seed: int, subset: int = 0) -> Dataset:
    if len(dataset) < 2:
        raise ValueError("At least two readable samples are required for evaluation")
    if val_split <= 0 or val_split >= 1:
        raise ValueError("val_split must be between 0 and 1")

    val_size = max(1, int(round(len(dataset) * val_split)))
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(seed)
    _, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size], generator=generator)
    if subset > 0 and subset < len(val_ds):
        return Subset(val_ds, list(range(subset)))
    return val_ds


def evaluate(
    data_dir: str,
    model_dir: str,
    filename_col: str = DEFAULT_FILENAME_COL,
    label_col: str = DEFAULT_LABEL_COL,
    val_split: float = 0.1,
    batch_size: int = 16,
    subset: int = 0,
    seed: int = 42,
    output_dir: str = "outputs/evaluation",
) -> dict[str, object]:
    data_path = Path(data_dir)
    model_path = Path(model_dir)
    dataset = CatDogEvalDataset(data_path / "labels.csv", data_path / "images", filename_col, label_col)
    val_ds = validation_subset(dataset, val_split=val_split, seed=seed, subset=subset)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = ViTImageProcessor.from_pretrained(model_path)
    model = ViTForImageClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()

    def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        images = [item["image"] for item in batch]
        labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
        inputs = processor(images=images, return_tensors="pt")
        inputs["labels"] = labels
        return inputs

    loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate)
    y_true: list[int] = []
    y_pred: list[int] = []

    with torch.no_grad():
        for batch in loader:
            labels = batch["labels"].to(device)
            inputs = {key: value.to(device) for key, value in batch.items() if key != "labels"}
            outputs = model(**inputs)
            preds = outputs.logits.argmax(dim=-1)
            y_true.extend(int(value) for value in labels.cpu().tolist())
            y_pred.extend(int(value) for value in preds.cpu().tolist())

    result = compute_classification_metrics(y_true, y_pred, labels=LABELS)
    metadata = {
        "created_at": utc_now_iso(),
        "checkpoint": str(model_path),
        "dataset": str(data_path),
        "filename_col": filename_col,
        "label_col": label_col,
        "split_policy": f"validation subset from val_split={val_split}",
        "batch_size": batch_size,
        "subset": subset,
        "seed": seed,
        "device": str(device),
    }
    artifact_paths = write_evaluation_artifacts(output_dir, result, metadata, labels=LABELS)
    payload = {
        "result": {
            "total": result.total,
            "correct": result.correct,
            "accuracy": result.accuracy,
            "confusion_matrix": result.confusion_matrix,
            "per_class": result.per_class,
        },
        "artifacts": artifact_paths,
    }
    print(json.dumps(payload, indent=2))
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a local ViT cats-vs-dogs checkpoint")
    parser.add_argument("--data-dir", default="data", help="Directory with labels.csv and images/")
    parser.add_argument("--model-dir", default="models/vit_catsdogs", help="Directory of saved model")
    parser.add_argument("--filename-col", default=DEFAULT_FILENAME_COL)
    parser.add_argument("--label-col", default=DEFAULT_LABEL_COL)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--subset", type=int, default=0, help="Evaluate only this many validation samples when >0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="outputs/evaluation")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(
        data_dir=args.data_dir,
        model_dir=args.model_dir,
        filename_col=args.filename_col,
        label_col=args.label_col,
        val_split=args.val_split,
        batch_size=args.batch_size,
        subset=args.subset,
        seed=args.seed,
        output_dir=args.output_dir,
    )
