from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from src.api import main
from src.api.main import app, decode_image


def test_health_endpoint_is_available_without_loading_model_dependencies():
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert "ready_for_prediction" in payload
    assert "artifact_policy" in payload


def test_upload_requires_supported_image_bytes():
    response = TestClient(app).post(
        "/predict", files={"file": ("sample.png", b"sample text", "image/png")}
    )
    assert response.status_code == 415


def test_upload_enforces_encoded_size_limit(monkeypatch):
    monkeypatch.setattr(main, "MAX_IMAGE_BYTES", 4)
    response = TestClient(app).post(
        "/predict", files={"file": ("sample.png", b"12345", "image/png")}
    )
    assert response.status_code == 413


def test_upload_enforces_total_body_limit_before_parsing(monkeypatch):
    monkeypatch.setattr(main, "MAX_REQUEST_BYTES", 32)
    response = TestClient(app).post(
        "/predict", files={"file": ("sample.png", b"12345", "image/png")}
    )
    assert response.status_code == 413


def test_upload_enforces_decoded_pixel_limit(monkeypatch):
    monkeypatch.setattr(main, "MAX_IMAGE_PIXELS", 3)
    data = BytesIO()
    with Image.new("RGB", (2, 2)) as source:
        source.save(data, format="PNG")

    response = TestClient(app).post(
        "/predict", files={"file": ("sample.png", data.getvalue(), "image/png")}
    )
    assert response.status_code == 413


def test_decode_accepts_supported_image():
    data = BytesIO()
    with Image.new("L", (2, 2)) as source:
        source.save(data, format="PNG")

    with decode_image(data.getvalue()) as image:
        assert image.mode == "RGB"
        assert image.size == (2, 2)
