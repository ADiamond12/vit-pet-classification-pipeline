"""Fine-tune a ViT checkpoint and save release-ready training metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset, Subset, random_split
from transformers import TrainingArguments, Trainer, ViTForImageClassification, ViTImageProcessor

from src.training.metrics import LABELS, count_labels, utc_now_iso, write_json

DEFAULT_FILENAME_COL = "image_name"
DEFAULT_LABEL_COL = "label"
LABEL2ID = {"cat": 0, "dog": 1}
ID2LABEL = {value: key for key, value in LABEL2ID.items()}


def filter_existing_images(df: pd.DataFrame, images_dir: str | Path, filename_col: str) -> pd.DataFrame:
    valid_mask: list[bool] = []
    missing = 0
    unreadable = 0
    images_path = Path(images_dir)

    for filename in df[filename_col]:
        image_path = images_path / str(filename)
        if not image_path.exists():
            missing += 1
            valid_mask.append(False)
            continue
        try:
            with Image.open(image_path) as image:
                image.verify()
            valid_mask.append(True)
        except (UnidentifiedImageError, OSError):
            unreadable += 1
            valid_mask.append(False)

    mask = pd.Series(valid_mask, index=df.index)
    kept = int(mask.sum())
    total = len(df)
    print(f"Keeping {kept} samples out of {total} (missing: {missing}, unreadable: {unreadable}).")
    return df[mask].reset_index(drop=True)


def limited_dataframe(
    df: pd.DataFrame,
    label_col: str,
    max_samples_per_class: int | None,
    seed: int,
) -> pd.DataFrame:
    if not max_samples_per_class:
        return df.reset_index(drop=True)

    parts = []
    for label in LABELS:
        subset = df[df[label_col].astype(str).str.lower() == label]
        parts.append(subset.sample(n=min(len(subset), max_samples_per_class), random_state=seed))
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)


class CatDogDataset(Dataset):
    def __init__(
        self,
        csv_path: str | Path,
        images_dir: str | Path,
        filename_col: str,
        label_col: str,
        max_samples_per_class: int | None = None,
        seed: int = 42,
    ) -> None:
        df = pd.read_csv(csv_path)
        df = filter_existing_images(df, images_dir, filename_col)
        df[label_col] = df[label_col].astype(str).str.strip().str.lower()
        unknown = sorted(set(df[label_col]) - set(LABEL2ID))
        if unknown:
            raise ValueError(f"Unsupported labels: {unknown}")

        self.data = limited_dataframe(df, label_col, max_samples_per_class, seed)
        self.images_dir = Path(images_dir)
        self.filename_col = filename_col
        self.label_col = label_col

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.data.iloc[index]
        image_path = self.images_dir / str(row[self.filename_col])
        label_name = str(row[self.label_col]).strip().lower()
        try:
            image = Image.open(image_path).convert("RGB")
        except (UnidentifiedImageError, OSError) as exc:
            raise ValueError(f"Image at '{image_path}' is unreadable") from exc
        return {"image": image, "label": LABEL2ID[label_name]}


def split_dataset(dataset: Dataset, val_split: float, seed: int) -> tuple[Subset, Subset]:
    if len(dataset) < 2:
        raise ValueError("At least two readable samples are required for train/validation split")
    if val_split <= 0 or val_split >= 1:
        raise ValueError("val_split must be between 0 and 1")

    val_size = max(1, int(round(len(dataset) * val_split)))
    train_size = len(dataset) - val_size
    if train_size <= 0:
        raise ValueError("Validation split leaves no training samples")

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=generator)
    return train_ds, val_ds


def build_collate_fn(processor: ViTImageProcessor):
    def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        images = [item["image"] for item in batch]
        labels = [item["label"] for item in batch]
        inputs = processor(images=images, return_tensors="pt")
        inputs["labels"] = torch.tensor(labels)
        return inputs

    return collate


def freeze_backbone(model: ViTForImageClassification) -> None:
    for name, parameter in model.named_parameters():
        if not name.startswith("classifier."):
            parameter.requires_grad = False


def train(
    data_dir: str,
    output_dir: str,
    filename_col: str = DEFAULT_FILENAME_COL,
    label_col: str = DEFAULT_LABEL_COL,
    val_split: float = 0.1,
    epochs: int = 1,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
    model_name: str = "google/vit-base-patch16-224-in21k",
    seed: int = 42,
    max_samples_per_class: int | None = None,
    freeze_encoder: bool = False,
    metadata_output: str | None = None,
) -> dict[str, object]:
    data_path = Path(data_dir)
    output_path = Path(output_dir)
    csv_path = data_path / "labels.csv"
    images_dir = data_path / "images"

    dataset = CatDogDataset(
        csv_path=csv_path,
        images_dir=images_dir,
        filename_col=filename_col,
        label_col=label_col,
        max_samples_per_class=max_samples_per_class,
        seed=seed,
    )
    train_ds, val_ds = split_dataset(dataset, val_split=val_split, seed=seed)

    processor = ViTImageProcessor.from_pretrained(model_name)
    model = ViTForImageClassification.from_pretrained(
        model_name,
        num_labels=len(LABEL2ID),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )
    if freeze_encoder:
        freeze_backbone(model)

    training_args = TrainingArguments(
        output_dir=str(output_path),
        eval_strategy="epoch",
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        save_strategy="epoch",
        logging_dir=str(output_path / "logs"),
        learning_rate=learning_rate,
        remove_unused_columns=False,
        seed=seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=build_collate_fn(processor),
        processing_class=processor,
    )

    train_result = trainer.train()
    eval_metrics = trainer.evaluate()

    output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_path)
    processor.save_pretrained(output_path)

    label_counts = count_labels(dataset.data[label_col].tolist())
    metadata: dict[str, object] = {
        "created_at": utc_now_iso(),
        "model_name": model_name,
        "output_dir": str(output_path),
        "data_dir": str(data_path),
        "filename_col": filename_col,
        "label_col": label_col,
        "labels": list(LABELS),
        "label_counts": label_counts,
        "train_size": len(train_ds),
        "validation_size": len(val_ds),
        "val_split": val_split,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "seed": seed,
        "max_samples_per_class": max_samples_per_class,
        "freeze_encoder": freeze_encoder,
        "train_metrics": train_result.metrics,
        "eval_metrics": eval_metrics,
        "artifact_policy": "Model files remain under ignored models/ unless released intentionally as a checked artifact.",
    }
    metadata_path = Path(metadata_output) if metadata_output else output_path / "training-metadata.json"
    write_json(metadata_path, metadata)
    print(json.dumps({"model_dir": str(output_path), "metadata": str(metadata_path)}, indent=2))
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a ViT cats-vs-dogs checkpoint")
    parser.add_argument("--data-dir", default="data", help="Directory containing labels.csv and images/")
    parser.add_argument("--output-dir", default="models/vit_catsdogs", help="Saved model and processor directory")
    parser.add_argument("--filename-col", default=DEFAULT_FILENAME_COL)
    parser.add_argument("--label-col", default=DEFAULT_LABEL_COL)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--model-name", default="google/vit-base-patch16-224-in21k")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples-per-class", type=int, default=None)
    parser.add_argument("--freeze-encoder", action="store_true", help="Train only the classifier head")
    parser.add_argument("--metadata-output", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        filename_col=args.filename_col,
        label_col=args.label_col,
        val_split=args.val_split,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        model_name=args.model_name,
        seed=args.seed,
        max_samples_per_class=args.max_samples_per_class,
        freeze_encoder=args.freeze_encoder,
        metadata_output=args.metadata_output,
    )
