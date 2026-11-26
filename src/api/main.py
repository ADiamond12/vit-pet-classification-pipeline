import io
import os
import warnings
from functools import lru_cache

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError
from starlette.types import ASGIApp, Receive, Scope, Send
from src.inference.bootstrap import DEFAULT_MODEL_DIR, MODEL_REPO_ID_ENV, checkpoint_status

MODEL_DIR = os.getenv("MODEL_DIR", DEFAULT_MODEL_DIR.as_posix())
MODEL_REPO_ID = os.getenv(MODEL_REPO_ID_ENV)
MAX_IMAGE_BYTES = 8 * 1024 * 1024
MAX_REQUEST_BYTES = MAX_IMAGE_BYTES + 64 * 1024
MAX_IMAGE_PIXELS = 16_000_000
IMAGE_FORMATS = ("JPEG", "PNG")

class LimitPredictionBody:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["method"] != "POST" or scope["path"].rstrip("/") != "/predict":
            await self.app(scope, receive, send)
            return

        length = dict(scope["headers"]).get(b"content-length")
        if length and length.isdigit() and (
            len(length) > len(str(MAX_REQUEST_BYTES)) or int(length) > MAX_REQUEST_BYTES
        ):
            response = JSONResponse({"detail": "Request exceeds the upload limit."}, status_code=413)
            await response(scope, receive, send)
            return

        received = 0

        async def limited_receive():
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > MAX_REQUEST_BYTES:
                    raise HTTPException(status_code=413, detail="Request exceeds the upload limit.")
            return message

        await self.app(scope, limited_receive, send)


app = FastAPI(title="ViT Pet Classification API")
app.add_middleware(LimitPredictionBody)


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
    contents = await file.read(MAX_IMAGE_BYTES + 1)
    if len(contents) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Image exceeds the upload limit.")
    with decode_image(contents) as image:
        try:
            from src.inference.predict import predict_image

            model, processor = get_model_bundle()
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail="Model is unavailable.") from exc
        label, conf = predict_image(image, model, processor)
    return {"label": label, "confidence": conf}


def decode_image(contents: bytes) -> Image.Image:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(contents), formats=IMAGE_FORMATS) as source:
                if source.width * source.height > MAX_IMAGE_PIXELS:
                    raise HTTPException(status_code=413, detail="Image exceeds the pixel limit.")
                if getattr(source, "is_animated", False):
                    raise HTTPException(status_code=415, detail="Animated images are unsupported.")
                source.verify()
            with Image.open(io.BytesIO(contents), formats=IMAGE_FORMATS) as source:
                return source.convert("RGB")
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise HTTPException(status_code=413, detail="Image exceeds the pixel limit.") from exc
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
        raise HTTPException(status_code=415, detail="Upload a valid JPEG or PNG image.") from exc
