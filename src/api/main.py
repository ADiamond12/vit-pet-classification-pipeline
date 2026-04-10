import io
import os
from functools import lru_cache

from fastapi import FastAPI, UploadFile, File
from PIL import Image
from src.inference.bootstrap import DEFAULT_MODEL_DIR, MODEL_REPO_ID_ENV
from src.inference.predict import load_model, predict_image

MODEL_DIR = os.getenv("MODEL_DIR", DEFAULT_MODEL_DIR.as_posix())
MODEL_REPO_ID = os.getenv(MODEL_REPO_ID_ENV)

# Simple FastAPI app exposing a single prediction endpoint
app = FastAPI(title="Cats vs Dogs API")


@lru_cache(maxsize=1)
def get_model_bundle():
    return load_model(MODEL_DIR, repo_id=MODEL_REPO_ID)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_dir": MODEL_DIR,
        "bootstrap_repo_id_configured": bool(MODEL_REPO_ID),
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read the uploaded image bytes and normalise to RGB
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    model, processor = get_model_bundle()
    label, conf = predict_image(image, model, processor)
    return {"label": label, "confidence": conf}
