from fastapi.testclient import TestClient


def test_ping(client: TestClient):
    response = client.get("/api/v1/ping")
    assert response.status_code == 200
    assert response.json() == {}
