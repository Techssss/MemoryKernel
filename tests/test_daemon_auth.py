import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from memk.server.daemon import app


def test_api_token_rejects_protected_route(monkeypatch):
    monkeypatch.setenv("MEMK_API_TOKEN", "secret-token")
    client = TestClient(app)

    response = client.get("/watcher/status")

    assert response.status_code == 401
    assert response.json()["detail"]["code"] == "auth_required"
    assert response.headers["X-Request-ID"]


def test_api_token_accepts_bearer_header(monkeypatch):
    monkeypatch.setenv("MEMK_API_TOKEN", "secret-token")
    client = TestClient(app)

    response = client.get(
        "/watcher/status",
        headers={"Authorization": "Bearer secret-token"},
    )

    assert response.status_code == 200
    assert response.json()["status"]["running"] is False
    assert response.headers["X-Request-ID"]


def test_health_remains_public_when_auth_enabled(monkeypatch):
    monkeypatch.setenv("MEMK_API_TOKEN", "secret-token")
    client = TestClient(app)

    response = client.get("/v1/health")

    assert response.status_code == 200
    assert response.json()["auth_enabled"] is True
