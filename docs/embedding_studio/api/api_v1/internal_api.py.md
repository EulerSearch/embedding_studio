# Documentation for `add_internal_endpoints`

## Functionality
The `add_internal_endpoints` function adds internal API endpoints to a given FastAPI router. It integrates endpoints for deletion, upsertion, reindexing, inference deployment, and vectordb. This function employs helper routers to register tasks and restrict available routes.

## Parameters

- `api_router`: A FastAPI APIRouter instance to which internal endpoints are attached.

## Usage

- **Purpose**: To group and register internal endpoints for task-based operations, enabling asynchronous processing.

### Example

Below is an example of how to use the `add_internal_endpoints` function:

```python
from fastapi import APIRouter
from embedding_studio.api.api_v1.internal_api import add_internal_endpoints

app_router = APIRouter()
add_internal_endpoints(app_router)
```