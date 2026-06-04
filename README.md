# ViT Pet Classification Pipeline

Computer-vision delivery path for cats-vs-dogs classification: public dataset preparation, Vision Transformer fine-tuning, evaluation artifacts, FastAPI inference, CLI prediction, and a Streamlit review UI.

![Streamlit UI empty state](assets/ui-empty.png)

## What this demonstrates

- ML delivery engineering, not only a notebook experiment.
- Separate dataset preparation, training, evaluation, inference, API, and UI modules.
- Public-repo posture: no datasets, model weights, credential files, or private artifacts committed.
- Tests that validate checkpoint resolution, API contract behavior, public UI boundaries, evaluation artifacts, and release-manifest checksums without requiring model weights.

## Portfolio role

This is a supporting ML-delivery project until a project-owned checkpoint is published with a reproducible evaluation artifact. The current public value is the engineering boundary around a vision model: how data is prepared, how a checkpoint is trained, how evaluation evidence is written, how inference is served, and how the UI behaves when a checkpoint is not present.

The repo now includes the full promotion path: Oxford-IIIT Pet binary dataset preparation, deterministic fine-tuning metadata, evaluation JSON, confusion matrix output, model-card templates, and a checksum manifest for GitHub Release assets. Model weights still stay out of normal git history.

For a bounded end-to-end candidate run, use `scripts/run_release_candidate.ps1`. It prepares a public subset, trains a local checkpoint, writes evaluation artifacts, and generates the release checksum manifest in ignored folders.

## Tech stack

- Hugging Face Transformers on PyTorch
- torchvision for the Oxford-IIIT Pet loader
- NumPy and pandas for data handling
- FastAPI for the inference endpoint
- Streamlit for the optional review UI
- pytest for lightweight public checks

## Repository posture

- `LICENSE` and `SECURITY.md` are included for public sharing.
- CI runs lightweight tests that do not require shipping model weights.
- Model artifacts, datasets, caches, and local outputs stay out of version control.
- The repo supports local checkpoints first and an optional first-run bootstrap path for a published fine-tuned model.

## Reviewer walkthrough

For a safe review without private datasets or model weights:

1. Read the public boundaries in `SECURITY.md`.
2. Run `pytest` to verify inference, API, UI-boundary, evaluation-artifact, and release-manifest behavior.
3. Inspect `src/training/prepare_oxford_pet_binary.py` for public dataset preparation.
4. Inspect `src/training/train_vit.py` and `src/training/eval_vit.py` for training metadata, eval JSON, and confusion matrix output.
5. Inspect `src/api/main.py`, `src/inference/predict.py`, and `src/ui/app.py` for delivery surfaces.
6. Run `powershell -ExecutionPolicy Bypass -File scripts\run_release_candidate.ps1` if you want to exercise the full local prepare-train-evaluate-manifest path.
7. Launch the UI with `streamlit run src/ui/app.py` to verify the clean upload flow.

Prediction screenshots are intentionally not shipped until a project-owned public checkpoint is published or a local checkpoint is trained in `models/vit_catsdogs`.

## Reviewer proof

- **Problem:** many ML demos publish a prediction screen without proving where the model, data, and metrics came from.
- **First command:** `pytest`
- **Proof artifact:** safe empty-state UI screenshot plus tests for checkpoint resolution, API contract behavior, eval artifact generation, and release-manifest checksums.
- **Candidate workflow:** `scripts/run_release_candidate.ps1` runs prepare, train, evaluate, and checksum manifest generation with safe local defaults.
- **Visual proof:** `assets/ui-empty.png` shows the review UI without implying an unavailable checkpoint.
- **Health proof:** `/health` reports whether a local or bootstrap-cached checkpoint is ready without loading weights or downloading anything.
- **Validation:** pytest tests and GitHub Actions CI cover checkpoint status, helper behavior, API contract, evaluation artifacts, release manifest checksums, and public UI boundary wording.
- **Current limitation:** prediction screenshots, confidence scores, and accuracy claims are withheld until a project-owned checkpoint and evaluation artifact are published.

## Next flagship step

To promote this from supporting project to a stronger ML portfolio project:

1. Prepare Oxford-IIIT Pet binary data with `src/training/prepare_oxford_pet_binary.py`.
2. Train or publish a project-owned checkpoint under ignored `models/`.
3. Regenerate metrics with `src/training/eval_vit.py`, which writes `eval.json`, `confusion_matrix.csv`, and `evaluation-report.md`.
4. Fill in `docs/model-card-template.md` and `docs/evaluation-template.md` with dataset source, label policy, limitations, and evaluation date.
5. Generate `outputs/release/model-release-manifest.json` and upload the checkpoint as a GitHub Release asset with checksum.
6. Capture one prediction screenshot from that owned checkpoint.
7. Keep the no-private-artifact rule: `data/`, `models/`, `.cache/`, and `outputs/` stay ignored unless a curated public artifact is intentionally released.

The RC50 local candidate is intentionally small and should remain workflow evidence unless the resulting metrics are strong enough for public model-quality presentation.

## Directory layout

```
vit-pet-classification-pipeline/
|- data/             # ignored: labels, images, prepared public dataset copies
|- models/           # ignored: saved fine-tuned model/processor
|- outputs/          # ignored: evaluation, release, and local run artifacts
|- scripts/          # release-candidate runner
|- src/
|  |- training/      # prepare data, inspect data, train, evaluate, release manifest
|  |- inference/     # bootstrap.py, predict.py
|  |- api/           # main.py
|  |- ui/            # app.py
|- assets/           # sanitized screenshots
|- docs/             # model card, evaluation template, demo notes
|- requirements.txt
|- .gitignore
`- README.md
```

## Installation

1. Clone the repo.
2. Create and activate a virtual environment:
   ```
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
   Optional editable install for local development:
   ```
   pip install -e .[dev]
   ```

## Public dataset preparation

For a reproducible cats-vs-dogs path, use Oxford-IIIT Pet through torchvision:

```
python src/training/prepare_oxford_pet_binary.py --root data/raw --output-dir data/oxford_pet_binary
```

This writes:

- `data/oxford_pet_binary/labels.csv`
- `data/oxford_pet_binary/images/`
- `data/oxford_pet_binary/dataset_manifest.json`

Oxford-IIIT Pet is documented by the University of Oxford and can be loaded through `torchvision.datasets.OxfordIIITPet` with `target_types="binary-category"`. Prepared data remains ignored by git.

## Usage

- Inspect data paths:
  ```
  python src/training/inspect_data.py --data-dir data/oxford_pet_binary
  ```
- Train the model:
  ```
  python src/training/train_vit.py --data-dir data/oxford_pet_binary --output-dir models/vit_catsdogs --epochs 1 --freeze-encoder
  ```
- Bootstrap from a published fine-tuned checkpoint at runtime:
  ```
  $env:VIT_PET_MODEL_REPO_ID="<published-fine-tuned-checkpoint>"
  uvicorn src.api.main:app --reload
  ```
- Evaluate a local checkpoint and write reviewer artifacts:
  ```
  python src/training/eval_vit.py --data-dir data/oxford_pet_binary --model-dir models/vit_catsdogs --output-dir outputs/evaluation
  ```
- Generate a checksum manifest before attaching model files to a GitHub Release:
  ```
  python src/training/release_manifest.py --artifact-dir models/vit_catsdogs --output outputs/release/model-release-manifest.json
  ```
- Run the bounded release-candidate workflow:
  ```
  powershell -ExecutionPolicy Bypass -File scripts\run_release_candidate.ps1
  ```
- Run the API:
  ```
  uvicorn src.api.main:app --reload
  ```
- Check API readiness without loading model weights:
  ```
  curl http://127.0.0.1:8000/health
  ```
- Launch the Streamlit UI:
  ```
  streamlit run src/ui/app.py
  ```
- CLI prediction:
  ```
  python src/inference/predict.py --image-path data/images/0.jpg
  ```

## Notes

- Defaults expect labels `cat` and `dog`; adjust flags if your CSV differs.
- The classifier head is reinitialized for 2 classes; the mismatched-size warning is expected.
- `data/`, `models/`, `.cache/`, `outputs/`, and `.venv/` are ignored to keep the repo lean.
- If `models/vit_catsdogs` is missing, the API and CLI can bootstrap from `VIT_PET_MODEL_REPO_ID`.
- This repo does not pin a public checkpoint by default; use the bootstrap env var only for a published fine-tuned model you explicitly want to trust.

## Testing

Run the current lightweight test suite with:

```
pytest
```

These tests cover the inference helper layer, API contract, public UI boundary, evaluation artifact generation, and release manifest checksums without requiring committed model weights, private datasets, or a published checkpoint.

## Demo screenshot

The committed screenshot is sanitized and shows the clean Streamlit upload flow without external company branding, private data, datasets, or model artifacts.

```
assets/ui-empty.png
```

Prediction screenshots should be regenerated only after a project-owned public checkpoint is published or after local training into `models/vit_catsdogs`.

## Public-readiness checklist

- [x] README, report, license, security notes
- [x] Tests pass without model weights
- [x] Dataset/model/cache/output folders ignored
- [x] Sanitized screenshots only
- [x] No credentials or private data committed
