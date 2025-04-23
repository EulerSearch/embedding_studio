# Documentation for Inference Deployment Tasks API

## Documentation for `deploy`

### Functionality
Deploys a model using a green deployment strategy. This endpoint creates a deployment task, sends it to a worker, and returns the result of the deployment process.

### HTTP Details
- **HTTP Method:** POST
- **Endpoint:** /deploy

### Parameters
- **body** (`ModelDeploymentRequest`): Contains the required details for model deployment.

### Output
Returns a `ModelDeploymentResponse` containing the outcome of the deployment process. If the task creation or processing fails, it raises an HTTP 500 error.

### Motivation
This endpoint is designed to handle model deployments by automating task creation and execution, ensuring a smooth deployment process.

### Example Usage
```
curl -X POST http://<host>/deploy \
  -H "Content-Type: application/json" \
  -d '{"model_id": "123", "config": {"param": "value"}}'
```

---

## Documentation for `get_deploy_task`

### Functionality
This endpoint retrieves the details of a specified deployment task. It uses the provided task_id to search for a deployment record and returns a `ModelDeploymentResponse` object if the task exists. In case the task is not found, it raises a 404 HTTP error.

### Parameters
- `task_id`: A string that uniquely identifies the deployment task to be retrieved.

### Usage
- **Purpose**: Fetch detailed information about a model deployment based on the specified task ID.
- **HTTP Method:** GET
- **Endpoint:** /deploy/{task_id}
- **Response:** Returns a `ModelDeploymentResponse` with task details or a 404 error if no matching task is found.

#### Example
```
curl -X GET "http://<server_address>/deploy/1234" \
     -H "Content-Type: application/json"
```

---

## Documentation for `get_model_deploy_status`

### Functionality
This API endpoint retrieves the deployment status for a specific embedding model by its ID. It accesses the internal task manager to check if a deployment task exists for the provided model and returns the task details if found.

### Parameters
- `embedding_model_id`: A string representing the unique ID of the model whose deployment status is to be retrieved.

### Usage
- **Purpose**: To obtain the status of a model deployment task.

#### Example
Use the following curl command to query the deployment status:
```bash
curl -X GET "http://<your-domain>/deploy-status/<embedding_model_id>"
```

### Response Schema
The response follows the structure of `ModelDeploymentResponse` which may include the following fields:
- `task_id`: A unique identifier for the deployment task.
- `status`: The current status of the deployment (e.g. pending, in progress, completed).
- `metadata`: Optional dictionary with additional context or information regarding the deployment process.

### Additional Notes
- The endpoint returns a 404 error if the deployment task for the given model ID is not found.
- The underlying task is retrieved using the context's `model_deployment_task.get_by_model_id` method.

---

## Documentation for `delete`

### Functionality
Deletes an embedding model via a deletion task. It creates a `ModelDeletionRequest`-based task, sends it to the deletion worker, and converts the result into a `ModelDeletionResponse`. If the task fails, an HTTPException is raised.

### Parameters
- `body`: `ModelDeletionRequest` containing the deletion parameters, including the model identifier required for removal.

### Usage
- **Purpose**: Initiate deletion of an embedding model from the inference service.

#### Example
Using curl:
```
curl -X POST "http://<api_host>/delete" \
     -H "Content-Type: application/json" \
     -d '{"model_id": "1234", "param": "value"}'
```

---

## Documentation for `get_delete_task`

### Functionality
Retrieves deletion task details for the given task ID. If a task exists, returns its details formatted as a `ModelDeletionResponse`. Otherwise, raises an HTTP 404 error indicating that the task cannot be found.

### Parameters
- `task_id` (str): The unique identifier of the deletion task to retrieve.

### Usage
- **Purpose**: This endpoint is used to fetch the status and details of a model deletion task that has been initiated.

#### API Endpoint Details
- **HTTP Method**: GET
- **Endpoint Path**: `/delete/{task_id}`
- **Input Format**: No request body is required.
- **Output Format**: A `ModelDeletionResponse` schema if the task exists, or an error response if not found.

#### Curl Example
```
curl -X GET "http://<host>/delete/your_task_id" \
     -H "accept: application/json"
```

---

## Documentation for `get_model_delete_status`

### Functionality
Retrieves deletion task details for a given embedding model ID. This endpoint uses a GET request to fetch the status of a model deletion process via the `/delete-status/{embedding_model_id}` path.

### Parameters
- `embedding_model_id`: Unique identifier for the model deletion task.

### Usage
- **Purpose**: To obtain the status of a model deletion task.
- **HTTP Method**: GET
- **Endpoint**: `/delete-status/{embedding_model_id}`
- **Response**: JSON object following the `ModelDeletionResponse` schema.

#### Example cURL Request
```
curl -X GET "http://<host>/delete-status/your_model_id" \
     -H "accept: application/json"
```