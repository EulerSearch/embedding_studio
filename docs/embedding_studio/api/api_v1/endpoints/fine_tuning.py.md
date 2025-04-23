## Documentation for Fine-Tuning Task Management

### `create_fine_tuning_task`

#### Functionality
Creates a new fine-tuning task based on the provided request body. It first retrieves the corresponding model iteration by the given `embedding_model_id`, validates the associated plugin, and checks for an existing task using an idempotency key. If an existing task is found, it returns that task; otherwise, a new task is created and dispatched via the fine-tuning worker.

#### Parameters
- `body`: A FineTuningTaskRunRequest object containing task details such as `embedding_model_id` and an optional `idempotency_key`.

#### Usage
- **Purpose**: Initiates a fine-tuning process by creating and sending a task that triggers model training or adjustments based on fine-tuning iteration data.

#### Example
Request payload:

```json
{
  "embedding_model_id": "model123",
  "idempotency_key": "unique-key-456"
}
```

Expected response:

```json
{
  "id": "task-id",
  "status": "PENDING"
}
```

---

### `get_fine_tuning_tasks`

#### Functionality
Retrieves a list of fine-tuning tasks from the backend with optional filtering. It supports pagination via offset and limit and allows filtering by task status.

#### Parameters
- `offset`: Number of tasks to skip.
- `limit`: Number of tasks to return.
- `status`: Optional task status filter to select specific tasks.

#### Usage
- **Purpose**: Fetch fine-tuning tasks to monitor and manage processing, pending, or completed tasks.

#### Example
A sample HTTP GET request:

```
GET /task?offset=0&limit=20&status=processing
```

This will return a list of fine-tuning tasks matching the criteria.

---

### `restart_fine_tuning_task`

#### Functionality
Restarts a fine-tuning task by resetting its status to pending. It checks for the existence of the task and its state. For tasks not in a processing state, it updates the status and dispatches the task with a new broker id.

#### Parameters
- `id`: Identifier of the task to be restarted.

#### Usage
- **Purpose**: Restart a fine-tuning task that might be stuck or failed, setting it to pending.

#### Example
```python
from embedding_studio.api.api_v1.endpoints.fine_tuning import restart_fine_tuning_task
result = restart_fine_tuning_task('TASK_ID')
```

---

### `cancel_fine_tuning_task`

#### Functionality
Cancels a fine-tuning task by aborting its ongoing execution using `dramatiq_abort` and updating its status to "canceled". It returns the updated task details on success.

#### Parameters
- `id`: A string representing the unique identifier of the task.

#### Usage
This endpoint is used to cancel a fine-tuning task. It retrieves the task by its `id`, aborts the task processing if it exists, updates its status, and returns the updated task information.

#### Example
Assuming a task with ID "12345" exists, a client can cancel it using:

```
PUT /task/12345/cancel
```

The response will include the task details with status set to "canceled".