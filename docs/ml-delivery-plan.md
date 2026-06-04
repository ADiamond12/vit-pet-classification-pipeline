# ViT ML Delivery Plan

This document defines the path from supporting ML packaging proof to a release-ready vision project. It is intentionally strict: no checkpoint, metric, confidence score, or prediction screenshot should be shown unless it came from this pipeline.

## Dataset

- Source: Oxford-IIIT Pet Dataset.
- Source URL: https://www.robots.ox.ac.uk/~vgg/data/pets/
- Loader: `torchvision.datasets.OxfordIIITPet`.
- Target: binary cats-vs-dogs label via `target_types="binary-category"`.
- License: Creative Commons Attribution-ShareAlike 4.0 International.
- Prepared data path: `data/oxford_pet_binary/`, ignored by git.

## Commands

Prepare a local binary dataset:

```
python src/training/prepare_oxford_pet_binary.py --root data/raw --output-dir data/oxford_pet_binary
```

Run a quick classifier-head pass:

```
python src/training/train_vit.py --data-dir data/oxford_pet_binary --output-dir models/vit_catsdogs --epochs 1 --freeze-encoder
```

Evaluate the checkpoint:

```
python src/training/eval_vit.py --data-dir data/oxford_pet_binary --model-dir models/vit_catsdogs --output-dir outputs/evaluation
```

Generate a release checksum manifest:

```
python src/training/release_manifest.py --artifact-dir models/vit_catsdogs --output outputs/release/model-release-manifest.json
```

Run a bounded release-candidate workflow with safe defaults:

```
powershell -ExecutionPolicy Bypass -File scripts\run_release_candidate.ps1
```

See `docs/release-candidate-runbook.md` for promotion gates and artifact policy.

## Required Public Artifacts

- `outputs/evaluation/eval.json`
- `outputs/evaluation/confusion_matrix.csv`
- `outputs/evaluation/evaluation-report.md`
- completed `docs/model-card-template.md`
- completed `docs/evaluation-template.md`
- `outputs/release/model-release-manifest.json`
- GitHub Release asset containing the checkpoint files and checksum manifest

## Release Boundary

Do not commit datasets, checkpoints, or local evaluation outputs to normal git history. Public release happens through a GitHub Release asset after checksums are generated and the model card is completed.

The RC50 workflow is useful engineering evidence because it exercises the complete path. It is not enough by itself for a public model-quality claim or prediction screenshot.
