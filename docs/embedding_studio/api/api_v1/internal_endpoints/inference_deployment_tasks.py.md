## Documentation

### Method: `deploy`

#### Functionality
Deploy a model using the green deployment strategy. This function receives deployment details and creates a model deployment task using the application context. It forwards the task to a background process.

#### Parameters
- **body**: The `ModelDeploymentRequest` containing model details and configuration for deploying a model.

#### Usage
- **Purpose**: Create and send a task for model deployment, then return a standardized response.

#### Example
**Request**:
```json
{
  "model_id": "abc123",
  "config": { ... }
}
```
**Response**:
```json
{
  "status": "success",
  "deployed_model_id": "abc123"
}
```

---

### Method: `get_deploy_task`

#### Functionality
Retrieves details of a specific deployment task using a task ID. Returns a `ModelDeploymentResponse` with task data if found. If not found, raises a HTTPException (404) error.

#### Parameters
- **task_id**: A string, unique ID of the deployment task.

#### Usage
- **Purpose**: Retrieve information on a deployment task.

#### Example
For task ID "123", a GET request to `/deploy/123` will return the deployment task details.

---

### Method: `get_model_deploy_status`

#### Functionality
This function retrieves the deployment details for a given embedding model. It obtains a deployment task based on the model ID and returns a formatted `ModelDeploymentResponse` if found. If no task is present, it raises an HTTPException.

#### Parameters
- **embedding_model_id**: A string representing the model's unique ID.

#### Usage
- **Purpose:** Retrieve deployment details for a specific embedding model using its unique identifier.

#### Example
A GET request to the endpoint `/deploy-status/{embedding_model_id}` returns the status for the specified model.

---

### Method: `delete`

#### Functionality
Delete a model by scheduling a deletion task. The function logs the request, creates a deletion task using the provided request body, and dispatches the task through a worker. On success, it returns the updated task details.

#### Parameters
- **body**: A `ModelDeletionRequest` object that contains the details required to delete the model.

#### Usage
- **Purpose**: To submit a deletion request and process the deletion asynchronously using a worker task.

#### Example
```python
from fastapi.testclient import TestClient
from embedding_studio.api.api_v1.internal_endpoints.inference_deployment_tasks import router

client = TestClient(router)
data = {
    "model_id": "your_model_id",
    "reason": "No longer needed"
}
response = client.post("/delete", json=data)
print(response.json())
```

---

### Method: `get_delete_task`

#### Functionality
Retrieves the details of a specific deletion task based on the provided task identifier. If found, returns the task details in a formatted response; otherwise, raises an HTTP 404 error.

#### Parameters
- **task_id**: The unique identifier of the deletion task.

#### Usage
- **Purpose**: To obtain detailed information about a deletion task.

#### Example
Assuming a task with ID "123", calling:
```python
get_delete_task("123")
```
will return the details of the deletion task if it exists, or throw a 404 error if not found.

---

### Method: `get_model_delete_status`

#### Functionality
Retrieves deletion task details using the model identifier. It accesses the deployment task context to fetch a deletion task and returns a formatted response. If no task is found, an HTTP exception is raised.

#### Parameters
- **embedding_model_id**: ID of the model for which the deletion task status is retrieved.

#### Usage
- **Purpose**: To obtain the current deletion request status for a model by providing its embedding model ID.

#### Example
```python
response = get_model_delete_status("model_id123")
print(response)
```