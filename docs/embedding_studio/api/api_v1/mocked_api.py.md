# Documentation for `add_mocked_endpoints`

## Functionality
Adds mocked endpoints to the provided FastAPI router. This function integrates mocked fine tuning endpoints into the API router, allowing simulated operations during testing or demonstration.

## Parameters
- `api_router`: An instance of FastAPI's APIRouter that manages the application routes.

## Usage
Use this function to extend a FastAPI router with endpoints specific to mocked fine tuning operations. The function attaches a sub-router with a preset URL prefix and tags.

### Example
```python
from fastapi import FastAPI, APIRouter
from embedding_studio.api.api_v1.mocked_api import add_mocked_endpoints

app = FastAPI()
router = APIRouter()
add_mocked_endpoints(router)
app.include_router(router)
```