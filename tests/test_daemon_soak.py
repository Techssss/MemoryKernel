import os
import uuid

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from memk.server.daemon import app


pytestmark = pytest.mark.skipif(
    os.getenv("MEMK_RUN_SOAK") != "1",
    reason="Set MEMK_RUN_SOAK=1 to run daemon soak tests.",
)


def test_daemon_health_soak(monkeypatch):
    monkeypatch.delenv("MEMK_API_TOKEN", raising=False)
    client = TestClient(app)

    for i in range(100):
        response = client.get("/v1/health", headers={"X-Request-ID": f"soak-{i}"})
        assert response.status_code == 200
        assert response.headers["X-Request-ID"] == f"soak-{i}"


def test_daemon_write_search_soak(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MEMK_API_TOKEN", raising=False)
    client = TestClient(app)
    workspace_id = f"soak-{uuid.uuid4().hex}"

    for i in range(25):
        response = client.post(
            "/v1/remember",
            json={
                "content": f"soak memory item {i} about storage growth",
                "importance": 0.5,
                "confidence": 1.0,
                "workspace_id": workspace_id,
            },
        )
        assert response.status_code == 200

    search = client.post(
        "/v1/search",
        json={
            "query": "storage growth",
            "limit": 5,
            "workspace_id": workspace_id,
        },
    )

    assert search.status_code == 200
    assert search.json()["data"]["results"]
    assert (tmp_path / "mem.db").exists()
