"""Utility helpers for offline single-image predictions."""

import argparse
import os
from typing import Union

import torch
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor

from src.inference.bootstrap import DEFAULT_MODEL_DIR, MODEL_REPO_ID_ENV, ensure_model_dir

MODEL_DIR = os.getenv("MODEL_DIR", DEFAULT_MODEL_DIR.as_posix())
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_dir: str = MODEL_DIR, repo_id: str | None = None):
    """Load a fine-tuned ViT model and its processor from disk or bootstrap cache."""
    resolved_model_dir = ensure_model_dir(model_dir=model_dir, repo_id=repo_id)
    processor = ViTImageProcessor.from_pretrained(resolved_model_dir)
    model = ViTForImageClassification.from_pretrained(resolved_model_dir)
    model.to(DEVICE)
    model.eval()
    return model, processor


def predict_image(image_input: Union[str, Image.Image], model, processor):
    """Run a single-image prediction from a filepath or PIL Image instance."""
    if isinstance(image_input, Image.Image):
        image = image_input
    else:
        image = Image.open(image_input).convert("RGB")

    inputs = processor(images=image, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)[0]
        pred_id = int(probs.argmax())
        conf = float(probs[pred_id])
    label = model.config.id2label[pred_id]
    return label, conf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-path", type=str, required=True, help="Path to the image to classify")
    parser.add_argument("--model-dir", type=str, default=MODEL_DIR, help="Local model directory")
    parser.add_argument(
        "--model-repo-id",
        type=str,
        default=os.getenv(MODEL_REPO_ID_ENV),
        help=f"Optional published fine-tuned checkpoint to bootstrap if --model-dir is missing (or set {MODEL_REPO_ID_ENV})",
    )
    args = parser.parse_args()

    model, processor = load_model(args.model_dir, repo_id=args.model_repo_id)
    label, conf = predict_image(args.image_path, model, processor)
    print(f"Prediction: {label} (confidence: {conf:.2%})")


if __name__ == "__main__":
    main()
