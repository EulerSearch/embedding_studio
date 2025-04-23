# Documentation for Internal Deletion Tasks API

---

## `POST /internal/deletion-tasks/run` — create_deletion_task

### Description
Creates a new deletion task that removes one or more object IDs from vector storage. If the specified `task_id` already exists, the existing task is returned instead.

### Request Fields
- `embedding_model_id` *(str)*: Identifier of the model whose vector index is targeted.
- `task_id` *(str, optional)*: Unique ID to make the operation idempotent.
- `object_ids` *(List[str])*: List of object IDs to delete.

### Request Example
```json
{
  "embedding_model_id": "embed-456",
  "task_id": "delete-xyz",
  "object_ids": ["obj-1", "obj-2"]
}
```

### Response Example
```json
{
  "id": "delete-xyz",
  "embedding_model_id": "embed-456",
  "status": "pending",
  "created_at": "2024-05-21T12:00:00Z",
  "updated_at": "2024-05-21T12:00:00Z",
  "failed_item_ids": []
}
```

---

## `GET /internal/deletion-tasks/info?task_id=...` — get_deletion_task

### Description
Fetches status, metadata, timestamps, and any failed deletions for a given deletion task. Useful for monitoring the progress of background object removal and determining retry targets.

### Response Example
```json
{
  "id": "delete-xyz",
  "embedding_model_id": "embed-456",
  "status": "done",
  "created_at": "2024-05-21T12:00:00Z",
  "updated_at": "2024-05-21T12:03:15Z",
  "failed_item_ids": [
    {
      "object_id": "obj-2",
      "detail": "Object not found in index"
    }
  ]
}
```

---

## `GET /internal/deletion-tasks/list` — list_deletion_tasks

### Description
Lists all internal deletion tasks with optional status filtering and pagination support.

---

## `PUT /internal/deletion-tasks/restart?task_id=...` — restart_deletion_task

### Description
Restarts a deletion task by setting its status to `pending` and resubmitting it to the task queue.

---

## `PUT /internal/deletion-tasks/cancel?task_id=...` — cancel_deletion_task

### Description
Cancels an active deletion task and marks it as canceled in the system.

---