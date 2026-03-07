"""API tests (health; liveness requires InsightFace installed)."""

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert "version" in data
    assert "environment" in data


def test_liveness_invalid_image():
    r = client.post("/api/v1/liveness", json={"image_base64": ""})
    assert r.status_code == 422  # validation error (empty string)


def test_liveness_bad_base64():
    r = client.post("/api/v1/liveness", json={"image_base64": "not-valid-base64!!"})
    # Decode can fail (400) or produce invalid image (200 with live=False)
    assert r.status_code in (400, 200)
    if r.status_code == 400:
        assert "detail" in r.json()
    else:
        assert "live" in r.json() and "confidence" in r.json()
