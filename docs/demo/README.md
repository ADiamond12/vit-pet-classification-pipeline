# ViT Pet Classification Pipeline Demo Storyboard

Use this storyboard for an honest public walkthrough. Do not show prediction confidence, accuracy, or model-output screenshots unless a project-owned checkpoint has been trained or published.

## 60-Second Reviewer Flow

1. Open the README and state the project role: supporting ML-delivery proof, not a flagship model benchmark yet.
2. Run `pytest` to show the public repo can validate helper and API contract behavior without private model artifacts.
3. Open `assets/ui-empty.png` or run `streamlit run src/ui/app.py` to show the clean upload/review surface.
4. Inspect `src/training/`, `src/inference/`, `src/api/`, and `src/ui/` to show the train/evaluate/serve/review separation.
5. Close with the promotion criteria: owned checkpoint, model card, evaluation artifact, then prediction screenshot.

## Screenshots To Capture

- `assets/ui-empty.png`: safe public UI state without private model weights.
- Future screenshot only after a project-owned checkpoint exists: one prediction result with model card and evaluation date linked nearby.

## What To Say

This project is useful as an ML-delivery boundary: it shows how model training, checkpoint resolution, FastAPI serving, CLI prediction, and Streamlit review would fit together while keeping private data and model artifacts out of the public repo.
