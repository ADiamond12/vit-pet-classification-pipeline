from fastapi.testclient import TestClient

from src.api.main import app


def test_health_endpoint_is_available_without_loading_model_dependencies():
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert "ready_for_prediction" in payload
    assert "artifact_policy" in payload
