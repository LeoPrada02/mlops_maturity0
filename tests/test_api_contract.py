import joblib
from fastapi.testclient import TestClient

from src.api.app import app, load_model


def test_health():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_sample_payload():
    client = TestClient(app)
    response = client.get("/sample-payload")
    assert response.status_code == 200
    body = response.json()
    assert "monthly_charge" in body
    assert "contract_type" in body
