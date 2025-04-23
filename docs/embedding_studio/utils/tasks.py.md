# Documentation for Task Management Functions

## `convert_to_response`

### Functionality
This function converts a database task object into a response model instance using the provided response schema. It leverages pydantic's `model_validate` method to ensure the output conforms to the response model.

### Parameters
- **`task`**: The database task object (BaseTaskInDb) that holds task data.
- **`response_schema`**: A pydantic model class, derived from BaseTaskResponse, to which the task is converted.

### Usage
Use this function to transform internal task records into API response formats that are validated against expected schemas.

#### Example
Assuming a valid task object and corresponding response model:
```python
response = convert_to_response(task, ResponseModel)
```

---

## `create_task_helpers_router`

### Functionality
Factory function that creates a FastAPI router with endpoints for managing tasks. It sets up endpoints for obtaining task information, listing tasks, restarting tasks, and cancelling tasks.

### Parameters
- **`task_crud`**: CRUD manager to perform task operations.
- **`response_model`**: Pydantic model used for API responses.
- **`worker_func`**: Function responsible for executing tasks.

### Usage
- **Purpose**: To quickly integrate task management API endpoints into a FastAPI application.

#### Example
```python
from fastapi import FastAPI
from embedding_studio.utils.tasks import create_task_helpers_router

# Assume task_crud, response_model, and worker_func are already defined and implemented
router = create_task_helpers_router(task_crud, response_model, worker_func)

app = FastAPI()
app.include_router(router)
```

---

## `_get_task`

### Functionality
The `_get_task` function retrieves a task from the database using its unique task ID. It then converts the task into a response model using the provided conversion function. If the task is not found, it raises an HTTP 404 error.

### Parameters
- **`task_id`**: A string representing the unique ID of the task.

### Usage
- **Purpose**: Retrieve a task and transform it into a response model for API endpoints.

#### Example
An example usage within a FastAPI endpoint:
```python
task = _get_task("example_id")
```
This returns the task as a response model if it exists, or raises a 404 error if not found.

---

## `get_task`

### Functionality
Fetches details of a specific task by its ID. The function retrieves the task via an internal helper, converts it into a defined response schema, and returns the task details. If the task does not exist, an HTTP 404 error is raised.

### Parameters
- **`task_id`**: A string representing the unique identifier of the task to be retrieved.

### Usage
- **Purpose**: Retrieve detailed information about a task using its unique ID.

#### Example
```
GET /info?task_id=<your_task_id>
```

---

## `get_tasks`

### Functionality
List tasks with optional filtering by status. Depending on provided parameters, it either retrieves all tasks or filters those based on the given status.

### Parameters
- **`offset`**: Number of items to skip (for pagination).
- **`limit`**: Maximum number of tasks to return.
- **`status`**: Optional task status to filter tasks.

### Usage
- **Purpose**: Retrieve a list of tasks with optional status filtering.

#### Example
Using a FastAPI client:
```python
response = client.get("/list", params={"offset": 0, "limit": 50, "status": "pending"})
tasks = response.json()
```

---

## `restart_task`

### Functionality
Restarts a task by its identifier. If the task is not processing, its status is set to pending and re-sent to the worker. Returns the updated task data as a response model, or raises an error if not found or restart fails.

### Parameters
- **`task_id`** (str): The unique identifier of the task to restart.

### Usage
- **Purpose**: To allow clients to reinitiate a task that is either not in processing mode or to recover from a failure.

#### Example
Assuming a REST API call:
```
PUT /restart?task_id="example_task_id"
```
On success, the API returns updated task information in the expected response model form.

---

## `cancel_task`

### Functionality
Cancel a task by ID. This endpoint aborts the task's execution in the message broker and updates its status to canceled, returning an updated task as a response model.

### Parameters
- **`task_id`**: The ID of the task to cancel.

### Usage
- **Purpose**: Cancel a running task and update its status to canceled.

#### Example
```python
cancel_task("example_task_id")
```