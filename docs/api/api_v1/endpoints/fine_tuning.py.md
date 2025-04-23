# Documentation for Fine-Tuning Task Management API

---

## `POST /fine-tuning/task` — create_fine_tuning_task

### Description
Creates a fine-tuning task using a batch of training data and optional metadata. It checks for duplicate tasks using the `idempotency_key`.

### Request Parameters
- `embedding_model_id` *(str)*: ID of the base embedding model.
- `batch_id` *(str, optional)*: Training data batch ID.
- `metadata` *(dict, optional)*: Custom metadata to associate with the task.
- `idempotency_key` *(str, optional)*: Unique key to prevent task duplication.
- `deploy_as_blue` *(bool, optional)*: Auto-deploy as active model after training.
- `wait_on_conflict` *(bool, optional)*: Wait if deployment conflicts occur.

### Request JSON Example
```json
{
  "embedding_model_id": "embed-123",
  "batch_id": "batch-xyz",
  "metadata": { "initiated_by": "admin" },
  "idempotency_key": "unique-key-abc",
  "deploy_as_blue": true,
  "wait_on_conflict": false
}
```

### Response JSON Example
```json
{
  "id": "task-001",
  "batch_id": "batch-xyz",
  "best_model_url": "https://storage/models/ft-001",
  "best_model_id": "model-ft-001",
  "metadata": { "initiated_by": "admin" },
  "idempotency_key": "unique-key-abc",
  "deploy_as_blue": true,
  "wait_on_conflict": false,
  "status": "pending"
}
```

---

## `GET /fine-tuning/task` — get_fine_tuning_tasks

### Description
Returns a list of fine-tuning tasks. Supports filtering by status and pagination.

### Query Parameters
- `offset` *(int)*: Number of tasks to skip (default is 0).
- `limit` *(int)*: Maximum number of tasks to return (default is 100).
- `status` *(TaskStatus, optional)*: Filter tasks by current status (`pending`, `processing`, `completed`, etc.).

### Response JSON Example
```json
[
  {
    "id": "task-001",
    "batch_id": "batch-xyz",
    "best_model_url": "https://storage/models/ft-001",
    "best_model_id": "model-ft-001",
    "metadata": { "initiated_by": "admin" },
    "status": "completed"
  }
]
```

---

## `PUT /fine-tuning/task/{id}/restart` — restart_fine_tuning_task

### Description
Restarts a task by resetting its status and pushing it to the task queue.

### Path Parameter
- `id` *(str)*: ID of the task to restart.

### Response JSON Example
```json
{
  "id": "task-001",
  "batch_id": "batch-xyz",
  "status": "pending"
}
```

---

## `PUT /fine-tuning/task/{id}/cancel` — cancel_fine_tuning_task

### Description
Cancels a running fine-tuning task. Terminates the associated background job and sets the task status to `"canceled"`.

### Path Parameter
- `id` *(str)*: ID of the task to cancel.

### Response JSON Example
```json
{
  "id": "task-001",
  "batch_id": "batch-xyz",
  "status": "canceled"
}
```

---