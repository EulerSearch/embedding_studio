# Documentation for `create_upsertion_task`

## Functionality

The `create_upsertion_task` method creates a new upsertion task and dispatches it to a worker. If a task with a given `task_id` already exists, the existing task is returned. If the specified task does not exist, a new task is created, sent to the worker, and the response is converted into an `UpsertionTaskResponse`.

## Parameters

- `body` (UpsertionTaskRunRequest): The request payload containing task details. If a `task_id` is provided and exists, the corresponding task is utilized. If not, a new task is created and dispatched.

## Usage

- **Purpose**: The method initiates an upsertion task for asynchronous processing.

### Example

Using FastAPI client:

```python
response = client.post(
    "/internal/upsertion-tasks/run",
    json=payload
)
print(response.json())
```