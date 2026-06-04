import json
from pathlib import Path

import pytest

from src.training.metrics import (
    compute_classification_metrics,
    compute_confusion_matrix,
    write_evaluation_artifacts,
)
from src.training.release_manifest import build_release_manifest


def test_confusion_matrix_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        compute_confusion_matrix([0, 1], [0])


def test_classification_metrics_include_accuracy_and_per_class_values():
    result = compute_classification_metrics([0, 0, 1, 1], [0, 1, 1, 1])

    assert result.total == 4
    assert result.correct == 3
    assert result.accuracy == 0.75
    assert result.confusion_matrix == [[1, 1], [0, 2]]
    assert result.per_class["cat"]["precision"] == 1.0
    assert result.per_class["cat"]["recall"] == 0.5
    assert result.per_class["dog"]["precision"] == pytest.approx(0.666667)
    assert result.per_class["dog"]["recall"] == 1.0


def test_evaluation_artifacts_write_json_csv_and_markdown(tmp_path: Path):
    result = compute_classification_metrics([0, 1], [0, 1])
    paths = write_evaluation_artifacts(
        tmp_path,
        result,
        metadata={"checkpoint": "models/vit_catsdogs", "dataset": "data/oxford_pet_binary"},
    )

    eval_json = Path(paths["eval_json"])
    matrix_csv = Path(paths["confusion_matrix"])
    report_md = Path(paths["evaluation_report"])

    assert eval_json.exists()
    assert matrix_csv.exists()
    assert report_md.exists()
    payload = json.loads(eval_json.read_text(encoding="utf-8"))
    assert payload["accuracy"] == 1.0
    assert "ViT Pet Classification Evaluation" in report_md.read_text(encoding="utf-8")
    assert "actual\\predicted" in matrix_csv.read_text(encoding="utf-8")


def test_release_manifest_records_checksums_without_including_unexpected_files(tmp_path: Path):
    artifact_dir = tmp_path / "models" / "vit_catsdogs"
    artifact_dir.mkdir(parents=True)
    checkpoint_dir = artifact_dir / "checkpoint-1"
    checkpoint_dir.mkdir()
    (artifact_dir / "config.json").write_text("{}", encoding="utf-8")
    (artifact_dir / "model.safetensors").write_text("weights", encoding="utf-8")
    (artifact_dir / "debug.log").write_text("ignore", encoding="utf-8")
    (checkpoint_dir / "model.safetensors").write_text("checkpoint", encoding="utf-8")

    manifest = build_release_manifest(
        artifact_dir=artifact_dir,
        output_path=tmp_path / "outputs" / "model-release-manifest.json",
        release_name="vit-catsdogs-test",
    )

    paths = [file["path"] for file in manifest["files"]]
    assert "config.json" in paths
    assert "model.safetensors" in paths
    assert "checkpoint-1/model.safetensors" not in paths
    assert "debug.log" not in paths
    assert all(len(file["sha256"]) == 64 for file in manifest["files"])
