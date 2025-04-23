# Documentation for Internal Upsertion Tasks API

---

## `POST /internal/upsertion-tasks/run` — create_upsertion_task

### Description
Creates a new internal upsertion task that inserts or updates embedding vectors using a specific deployed model.

### Request Fields
- `embedding_model_id` *(str)*: ID of the deployed embedding model to use.
- `task_id` *(str, optional)*: Optional unique ID for idempotency.
- `items` *(List[DataItem])*:
  - `object_id` *(str)*: Unique identifier of the item.
  - `payload` *(dict, optional)*: Input data or metadata for vectorization.
  - `item_info` *(dict, optional)*: Optional internal metadata.

### Request Example
```json
{
  "embedding_model_id": "embed-123",
  "task_id": "upsert-001",
  "items": [
    {
      "object_id": "doc-1",
      "payload": { "text": "neural nets basics" }
    }
  ]
}
```

### Response Example
```json
{
  "id": "upsert-001",
  "embedding_model_id": "embed-123",
  "status": "pending",
  "created_at": "2024-05-21T10:00:00Z",
  "updated_at": "2024-05-21T10:00:00Z",
  "failed_items": []
}
```

---

## `GET /internal/upsertion-tasks/info?task_id=...` — get_upsertion_task

### Description
Fetches full metadata, processing status, creation/update timestamps, and error reports (if any) for the specified upsertion task. This endpoint is useful for monitoring long-running or background vector ingestion jobs, and supports retry decision-making by providing detailed error messages for failed items.

---

## `GET /internal/upsertion-tasks/list` — list_upsertion_tasks

### Description
Paginated list of all internal upsertion tasks.

### Query Parameters
- `offset` *(int)*: Number of items to skip.
- `limit` *(int)*: Maximum number of tasks to return.
- `status` *(TaskStatus)*: Filter tasks by their current state.

---

## `PUT /internal/upsertion-tasks/restart?task_id=...` — restart_upsertion_task

### Description
Restarts a task by setting its status to `pending` and resending it to the worker queue.

---

## `PUT /internal/upsertion-tasks/cancel?task_id=...` — cancel_upsertion_task

### Description
Cancels an internal upsertion task in progress.

---