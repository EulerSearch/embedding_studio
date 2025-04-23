# Merged Documentation

## Documentation for `stub_broker`

### Functionality

The `stub_broker` fixture resets the state of the Redis broker by flushing all keys. It then returns this clean broker instance for use in tests.

### Parameters

N/A

### Usage

- **Purpose** - to provide a clean Redis broker instance for test functions.

#### Example

In your test module, you might write:

```python
def test_broker(stub_broker):
    broker = stub_broker
    # Proceed with tests using broker
```

---

## Documentation for `stub_worker`

### Functionality

The `stub_worker` fixture creates a worker to run tasks via the Redis broker. It starts the worker before tests and stops it after tests have run, ensuring a clean execution environment.

### Parameters

This fixture does not accept any parameters.

### Usage

This fixture is used in tests to provide a reliable worker instance for processing tasks. It handles the lifecycle of the worker automatically.

#### Example

```python
def test_worker(stub_worker):
    assert stub_worker is not None
```

---

## Documentation for `client_fixture`

### Functionality

The `client_fixture` provides a test client for the FastAPI application by instantiating `TestClient` using the main app. It allows testing of API endpoints. After tests, it cleans up by dropping the "fine_tuning" collection from the MongoDB database.

### Parameters

This fixture takes no parameters.

### Usage

Use this fixture in tests to send requests to API endpoints.

#### Example

```python
def test_api_status(client):
    response = client.get("/api/status")
    assert response.status_code == 200
```