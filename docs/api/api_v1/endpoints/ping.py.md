# Documentation for ping

## Functionality

This endpoint is a simple health check. It returns a JSON response and HTTP 200 status when the service is online.

## Parameters

None.

## Usage

- **Purpose**: Validate that the API back-end is running.

### Example
```bash
curl -X GET "http://<HOST>/ping"
```