# vit-pet-classification-pipeline - Cats vs Dogs Classification

Compact end-to-end ML project: data handling, Vision Transformer fine-tuning, FastAPI inference, and an optional Streamlit UI.

## Tech stack
- Hugging Face Transformers (ViT) on PyTorch
- NumPy and pandas for data handling
- FastAPI for the inference endpoint
- Streamlit for the optional UI
- Git for version control

## Repository posture

- `LICENSE` and `SECURITY.md` are included for public sharing.
- CI runs lightweight tests that do not require shipping model weights.
- Model artifacts and datasets stay out of version control on purpose.
- The repo supports local checkpoints first and an optional first-run bootstrap path for a published fine-tuned model.

## Directory Layout
```
vit-pet-classification-pipeline/
|- data/             # (ignored) labels.csv and images/
|- models/           # (ignored) saved fine-tuned model/processor
|- src/
|  |- training/      # inspect_data.py, train_vit.py, eval_vit.py
|  |- inference/     # bootstrap.py, predict.py (CLI helper)
|  |- api/           # main.py (FastAPI /predict)
|  |- ui/            # app.py (Streamlit UI)
|- assets/           # screenshots (add your images here)
|- requirements.txt
|- .gitignore
`- README.md
```

## Installation
1. Clone or extract into `vit-pet-classification-pipeline`.
2. (Recommended) Create and activate a virtual env:
   ```
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1   # PowerShell
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
   Optional editable install for local development:
   ```
   pip install -e .[dev]
   ```
4. Choose one model path:
   - local training path: train into `models/vit_catsdogs`
   - bootstrap path: set `VIT_PET_MODEL_REPO_ID` to a published fine-tuned checkpoint so the API/CLI can download it on first run
5. Add your dataset under `data/` if you plan to train locally:
   - `data/labels.csv` with filename and label columns (defaults: `image_name`, `label`)
   - `data/images/` containing the referenced images

## Usage
- Inspect data paths:
  ```
  python src/training/inspect_data.py
  ```
- Train the model:
  ```
  python src/training/train_vit.py --data-dir data --output-dir models/vit_catsdogs
  ```
- Or bootstrap from a published fine-tuned checkpoint at runtime:
  ```
  $env:VIT_PET_MODEL_REPO_ID="<published-fine-tuned-checkpoint>"
  uvicorn src.api.main:app --reload
  ```
- Evaluate validation accuracy:
  ```
  python src/training/eval_vit.py --data-dir data --model-dir models/vit_catsdogs
  ```
- Run the API:
  ```
  uvicorn src.api.main:app --reload
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
- `data/`, `models/`, `.cache/`, and `.venv/` are ignored to keep the repo lean.
- If `models/vit_catsdogs` is missing, the API and CLI can bootstrap from `VIT_PET_MODEL_REPO_ID`.
- This repo does not pin a public checkpoint by default; use the bootstrap env var only for a published fine-tuned model you explicitly want to trust.

## Testing

Run the current lightweight test suite with:

```
pytest
```

These tests cover the inference helper layer without requiring committed model weights.

## Screenshots
Place images in `assets/` and reference them here. Examples:
- Empty state: `assets/ui-empty.png`
- Cat prediction: `assets/ui-cat-1.png`
- Cat prediction (alt): `assets/ui-cat-2.png`
- Dog prediction: `assets/ui-dog-1.png`

Markdown example:
```
![Streamlit UI - empty](assets/ui-empty.png)
![Streamlit UI - cat](assets/ui-cat-1.png)
![Streamlit UI - dog](assets/ui-dog-1.png)
```

## GitHub remote
If you renamed the repo on GitHub, update the local remote:
```
git remote set-url origin https://github.com/<you>/vit-pet-classification-pipeline.git
```



