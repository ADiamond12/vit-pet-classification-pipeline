"""Evaluation artifact helpers for the public ML delivery path."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence


LABELS = ("cat", "dog")


@dataclass(frozen=True)
class EvaluationResult:
    total: int
    correct: int
    accuracy: float
    confusion_matrix: list[list[int]]
    per_class: dict[str, dict[str, float]]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def compute_confusion_matrix(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    label_count: int = 2,
) -> list[list[int]]:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")

    matrix = [[0 for _ in range(label_count)] for _ in range(label_count)]
    for true_label, pred_label in zip(y_true, y_pred):
        if true_label < 0 or true_label >= label_count:
            raise ValueError(f"true label {true_label} is outside 0..{label_count - 1}")
        if pred_label < 0 or pred_label >= label_count:
            raise ValueError(f"predicted label {pred_label} is outside 0..{label_count - 1}")
        matrix[true_label][pred_label] += 1
    return matrix


def compute_classification_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    labels: Sequence[str] = LABELS,
) -> EvaluationResult:
    matrix = compute_confusion_matrix(y_true, y_pred, label_count=len(labels))
    total = len(y_true)
    correct = sum(matrix[index][index] for index in range(len(labels)))
    per_class: dict[str, dict[str, float]] = {}

    for index, label in enumerate(labels):
        true_positive = matrix[index][index]
        predicted_as_class = sum(row[index] for row in matrix)
        actual_class = sum(matrix[index])

        precision = true_positive / predicted_as_class if predicted_as_class else 0.0
        recall = true_positive / actual_class if actual_class else 0.0
        per_class[label] = {
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "support": float(actual_class),
        }

    accuracy = correct / total if total else 0.0
    return EvaluationResult(
        total=total,
        correct=correct,
        accuracy=round(accuracy, 6),
        confusion_matrix=matrix,
        per_class=per_class,
    )


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: str | Path, payload: Mapping[str, object]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_confusion_matrix_csv(
    path: str | Path,
    matrix: Sequence[Sequence[int]],
    labels: Sequence[str] = LABELS,
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["actual\\predicted", *labels])
        for label, row in zip(labels, matrix):
            writer.writerow([label, *row])


def render_evaluation_markdown(
    metadata: Mapping[str, object],
    result: EvaluationResult,
    labels: Sequence[str] = LABELS,
) -> str:
    lines = [
        "# ViT Pet Classification Evaluation",
        "",
        "This report is generated from an explicit checkpoint and dataset split. Do not reuse it for a different checkpoint.",
        "",
        "## Run Metadata",
        "",
    ]
    for key, value in metadata.items():
        lines.append(f"- {key}: {value}")

    lines.extend(
        [
            "",
            "## Results",
            "",
            f"- Total examples: {result.total}",
            f"- Correct predictions: {result.correct}",
            f"- Accuracy: {result.accuracy:.4f}",
            "",
            "## Class Metrics",
            "",
            "| Class | Precision | Recall | Support |",
            "| --- | ---: | ---: | ---: |",
        ]
    )

    for label in labels:
        class_metrics = result.per_class[label]
        lines.append(
            f"| {label} | {class_metrics['precision']:.4f} | "
            f"{class_metrics['recall']:.4f} | {int(class_metrics['support'])} |"
        )

    lines.extend(
        [
            "",
            "## Confusion Matrix",
            "",
            "| Actual \\ Predicted | " + " | ".join(labels) + " |",
            "| --- | " + " | ".join("---:" for _ in labels) + " |",
        ]
    )
    for label, row in zip(labels, result.confusion_matrix):
        lines.append(f"| {label} | " + " | ".join(str(value) for value in row) + " |")

    lines.extend(
        [
            "",
            "## Release Decision",
            "",
            "- Add a prediction screenshot only when this report is linked to a project-owned checkpoint.",
            "- Keep data and model artifacts outside normal git history.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_evaluation_artifacts(
    output_dir: str | Path,
    result: EvaluationResult,
    metadata: Mapping[str, object],
    labels: Sequence[str] = LABELS,
) -> dict[str, str]:
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    eval_json = target_dir / "eval.json"
    matrix_csv = target_dir / "confusion_matrix.csv"
    report_md = target_dir / "evaluation-report.md"

    payload = {
        "metadata": dict(metadata),
        "labels": list(labels),
        "total": result.total,
        "correct": result.correct,
        "accuracy": result.accuracy,
        "confusion_matrix": result.confusion_matrix,
        "per_class": result.per_class,
    }
    write_json(eval_json, payload)
    write_confusion_matrix_csv(matrix_csv, result.confusion_matrix, labels)
    report_md.write_text(render_evaluation_markdown(metadata, result, labels), encoding="utf-8")

    return {
        "eval_json": str(eval_json),
        "confusion_matrix": str(matrix_csv),
        "evaluation_report": str(report_md),
    }


def count_labels(labels: Iterable[str]) -> dict[str, int]:
    counts = {label: 0 for label in LABELS}
    for label in labels:
        normalized = str(label).strip().lower()
        if normalized not in counts:
            raise ValueError(f"unsupported label '{label}'")
        counts[normalized] += 1
    return counts
