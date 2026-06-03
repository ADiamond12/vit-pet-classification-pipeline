import io
import os
from functools import lru_cache

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from src.inference.bootstrap import DEFAULT_MODEL_DIR, MODEL_REPO_ID_ENV, checkpoint_status

MODEL_DIR = os.getenv("MODEL_DIR", DEFAULT_MODEL_DIR.as_posix())
MODEL_REPO_ID = os.getenv(MODEL_REPO_ID_ENV)

app = FastAPI(title="ViT Pet Classification API")


@lru_cache(maxsize=1)
def get_model_bundle():
    try:
        from src.inference.predict import load_model
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Model dependencies are not installed. Install the project dependencies before calling /predict."
        ) from exc

    return load_model(MODEL_DIR, repo_id=MODEL_REPO_ID)


@app.get("/health")
def health():
    status = checkpoint_status(MODEL_DIR, MODEL_REPO_ID)
    return {
        "status": "ok",
        **status,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    try:
        from src.inference.predict import predict_image

        model, processor = get_model_bundle()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    label, conf = predict_image(image, model, processor)
    return {"label": label, "confidence": conf}
