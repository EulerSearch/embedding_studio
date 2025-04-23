# Documentation for ping

## Functionality

This endpoint serves as a health check. It returns an empty JSON object if the server is active.

## Parameters

None.

## Usage

- **Purpose**: To verify that the server is active and healthy.

### Example

```bash
curl -X GET http://localhost:8000/ping
```