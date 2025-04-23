# Documentation for Deletion Tasks API

---

## `POST /embeddings/deletion-tasks/run` — create_deletion_task

### Description
Creates a new deletion task for object IDs in the main vector database.

### Request Parameters
- `task_id` *(str, optional)*: Unique task identifier for deduplication.
- `object_ids` *(List[str])*: List of object IDs to delete from the index.

### Request JSON Example
```json
{
  "task_id": "task-abc123",
  "object_ids": ["obj-1", "obj-2"]
}
```

### Response JSON Example
```json
{
  "id": "task-abc123",
  "status": "pending",
  "created_at": "2024-05-21T14:00:00Z",
  "updated_at": "2024-05-21T14:00:00Z",
  "failed_item_ids": []
}
```

---

## `POST /embeddings/deletion-tasks/categories/run` — create_categories_deletion_task

### Description
Identical to `/run`, but deletes items from the categories vector index instead.

---

## `GET /embeddings/deletion-tasks/info?task_id=...` — get_deletion_task

### Description
Retrieves status and metadata about a specific deletion task.

### Query Parameters
- `task_id`: The ID of the deletion task.

---

## `GET /embeddings/deletion-tasks/list` — list_deletion_tasks

### Description
Returns paginated list of deletion tasks with optional status filter.

### Query Parameters
- `offset`: Items to skip for pagination.
- `limit`: Max number of results.
- `status`: Optional filter (`pending`, `done`, `canceled`, etc.)

### Response Example
```json
[
  {
    "id": "task-abc123",
    "status": "done",
    "created_at": "2024-05-21T14:00:00Z",
    "updated_at": "2024-05-21T14:10:00Z",
    "failed_item_ids": []
  }
]
```

---

## `PUT /embeddings/deletion-tasks/restart?task_id=...` — restart_deletion_task

### Description
Resets and re-queues a deletion task if not already processing.

---

## `PUT /embeddings/deletion-tasks/cancel?task_id=...` — cancel_deletion_task

### Description
Cancels a deletion task in progress and updates its status.

---