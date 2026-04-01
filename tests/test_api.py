"""
Production API Tests
"""

import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "running"


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_metrics():
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "requests_total" in response.json()


def test_analyze_invalid():
    response = client.post("/api/v1/analyze", json={})
    assert response.status_code == 422


def test_query_invalid():
    response = client.post("/api/v1/query", json={})
    assert response.status_code == 422


def test_security_headers():
    response = client.get("/")
    assert "X-Content-Type-Options" in response.headers


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
