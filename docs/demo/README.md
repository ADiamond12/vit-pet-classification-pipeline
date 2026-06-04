# ViT Pet Classification Pipeline Demo Storyboard

Use this storyboard for an honest public walkthrough. Do not show prediction confidence, accuracy, or model-output screenshots unless a project-owned checkpoint has been trained or published.

## 60-Second Reviewer Flow

1. Open the README and state the project role: supporting ML-delivery proof, not a flagship model benchmark yet.
2. Run `pytest` to show the public repo can validate helper, API, evaluation artifact, and release-manifest behavior without private model artifacts.
3. Run the FastAPI app and open `/health` to show checkpoint readiness without loading weights.
4. Open `assets/ui-empty.png` or run `streamlit run src/ui/app.py` to show the clean upload/review surface.
5. Inspect `src/training/prepare_oxford_pet_binary.py`, `src/training/train_vit.py`, and `src/training/eval_vit.py` to show the public dataset, training metadata, and evaluation-report path.
6. Inspect `src/inference/`, `src/api/`, and `src/ui/` to show the serve/review separation.
7. Close with the promotion criteria: owned checkpoint, model card, evaluation artifact, release checksum manifest, then prediction screenshot.

## Screenshots To Capture

- `assets/ui-empty.png`: safe public UI state without private model weights.
- Future screenshot only after a project-owned checkpoint exists: one prediction result with model card and evaluation date linked nearby.

## What To Say

This project is useful as an ML-delivery boundary: it shows how model training, checkpoint resolution, FastAPI serving, CLI prediction, and Streamlit review would fit together while keeping private data and model artifacts out of the public repo.

The key proof is not a prediction screenshot yet. The useful proof is that the repo can expose train/evaluate/API/UI surfaces, start safely without a checkpoint, and produce reproducible evaluation and release artifacts when a project-owned checkpoint exists.
