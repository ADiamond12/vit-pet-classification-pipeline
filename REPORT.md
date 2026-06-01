# vit-pet-classification-pipeline - Project Report

## Objective
Deliver a compact end-to-end pipeline to classify cat vs dog images: data prep, Vision Transformer fine-tuning, and serving predictions via API and UI. The goal is solid ML engineering across preprocessing, training, inference, and deployment.

Public portfolio status: supporting ML-delivery project. The repo proves the training-serving-review structure, but it intentionally avoids public prediction claims until an owned checkpoint and reproducible metrics artifact are released.

## Dataset & Preprocessing
- Layout: `data/labels.csv` (`image_name`, `label`), images in `data/images/`.
- Safety: `filter_existing_images` drops missing or unreadable files and reports counts (~24,290 kept of 25,000).
- Labels: `cat -> 0`, `dog -> 1`; normalized to lowercase and trimmed.

## Vision Transformer & Model Choice
The Vision Transformer (ViT) splits an image into fixed patches, embeds them with positional encodings, and processes the sequence with a Transformer encoder. Self-attention captures global relationships better than purely local filters and adapts well from a pretrained checkpoint with limited task data. We chose `google/vit-base-patch16-224` for its size/accuracy balance and replaced the head with a 2-class layer (`ignore_mismatched_sizes=True`).

## Training Setup & Evaluation Notes
- Hyperparameters: `epochs=1`, `batch_size=8`, `learning_rate=2e-5`, `val_split=0.1`, `save_strategy="epoch"`.
- Processing: `ViTImageProcessor` handles resize to 224x224, normalization, tensor conversion.
- Local evaluation path:
  ```
  python src/training/eval_vit.py --data-dir data --model-dir models/vit_catsdogs
  ```
- No public accuracy number is quoted in this repository because the public release does not ship the dataset, model weights, or a reproducible metrics artifact. Treat the evaluation command as the source of truth for any local checkpoint you train or explicitly bootstrap.
- Notes: Warnings about reinitialized classifier weights and `pin_memory` on CPU are expected and harmless.

## Inference & Serving
- CLI: `python src/inference/predict.py --image-path <path>` (loads from `models/vit_catsdogs` or an optional bootstrap cache).
- API: FastAPI (`src/api/main.py`) exposes `/predict` for image uploads -> JSON and `/health` for basic runtime status.
- UI: Streamlit (`src/ui/app.py`) uploads an image, calls the API, and shows the prediction.
- Bootstrap path: if no local checkpoint exists, set `VIT_PET_MODEL_REPO_ID` to a published fine-tuned checkpoint and the CLI/API can download it into `.cache/vit-pet-models/` on first run.

## Structure & Usage
- `src/training`: data inspection and ViT fine-tuning.
- `src/inference`: local prediction helper.
- `src/api`: FastAPI backend for real-time inference.
- `src/ui`: Streamlit front-end for manual testing.
- Ignored: `data/`, `models/`, `.cache/`, `.venv/` (see `.gitignore`).

Run:
1. `python -m venv .venv` and activate it.
2. `pip install -r requirements.txt`.
3. Place dataset under `data/`.
4. Inspect: `python src/training/inspect_data.py`.
5. Train: `python src/training/train_vit.py --data-dir data --output-dir models/vit_catsdogs`.
6. Optional bootstrap alternative: set `VIT_PET_MODEL_REPO_ID=<published-fine-tuned-checkpoint>` if you want automatic first-run download instead of a local model directory.
7. API: `uvicorn src.api.main:app --reload`.
8. UI: `streamlit run src/ui/app.py`.
9. CLI: `python src/inference/predict.py --image-path data/images/0.jpg`.

## Observations & Future Work
- Public metrics should be regenerated from a project-owned checkpoint before being quoted externally.
- Modular layout simplifies testing and extension.
- Future: more epochs/tuning, augmentation, metrics logging, early stopping, publishing a project-owned fine-tuned checkpoint, adding a model card, adding a dated evaluation artifact, and optional image-verification toggle for huge datasets.

## Delivery Summary
The project ties together data validation, ViT fine-tuning, FastAPI serving, CLI prediction, and a Streamlit review UI. The main engineering value is the separation between training, inference, API, and UI code, plus a public-repo posture that keeps datasets and model artifacts outside version control. Next steps are richer evaluation, broader tuning/augmentation, a project-owned public checkpoint, and broader tests around API/UI behavior.

