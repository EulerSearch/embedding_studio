# Documentation for test_ping

## Functionality

The test_ping function checks the ping endpoint. It sends a GET request to "/api/v1/ping" using a FastAPI TestClient. The function asserts that the status code is 200 and the JSON response is an empty object, ensuring the endpoint is active.

## Parameters

- **client**: A FastAPI TestClient instance for API testing.

## Usage

- **Purpose**: Validates that the /api/v1/ping endpoint is operational.

### Example

```python
def test_ping(client: TestClient):
    response = client.get("/api/v1/ping")
    assert response.status_code == 200
    assert response.json() == {}
```