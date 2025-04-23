# Documentation for Upsertion Tasks API

---

## `POST /embeddings/upsertion-tasks/run` — create_upsertion_task

### Description
Creates a new task to insert or update vector records in the default vector collection.

### Request Parameters
- `task_id` *(str, optional)*: Custom ID to ensure idempotent task creation.
- `items` *(List[DataItem])*: The list of vector documents to upsert.

Each `DataItem` has:
- `object_id` *(str)*: Unique identifier of the item.
- `payload` *(dict, optional)*: Structured metadata or content.
- `item_info` *(dict, optional)*: Internal metadata (e.g. file refs, source).

### Request Example
```json
{
  "task_id": "task-123",
  "items": [
    {
      "object_id": "obj-1",
      "payload": {
        "text": "deep learning paper"
      }
    }
  ]
}
```

### Response Example
```json
{
  "id": "task-123",
  "status": "pending",
  "created_at": "2024-05-20T12:00:00Z",
  "updated_at": "2024-05-20T12:00:00Z",
  "failed_items": []
}
```

---

## `POST /embeddings/upsertion-tasks/categories/run` — create_categories_upsertion_task

### Description
Same as `/run`, but uses the *categories vector database* instead of the default one.

---

## `GET /embeddings/upsertion-tasks/info?task_id=...` — get_upsertion_task

### Description
Fetches metadata and current status of an upsertion task.

### Query Parameters
- `task_id` *(str)*: The task ID to look up.

---

## `GET /embeddings/upsertion-tasks/list` — list_upsertion_tasks

### Description
Returns a paginated list of upsertion tasks.

### Query Parameters
- `offset` *(int)*: Number of tasks to skip.
- `limit` *(int)*: Max number of tasks to return.
- `status` *(TaskStatus, optional)*: Optional filter (`pending`, `processing`, etc.).

### Response Example
```json
[
  {
    "id": "task-123",
    "status": "done",
    "created_at": "2024-05-20T12:00:00Z",
    "updated_at": "2024-05-20T12:10:00Z",
    "failed_items": []
  }
]
```

---

## `PUT /embeddings/upsertion-tasks/restart?task_id=...` — restart_upsertion_task

### Description
Restarts an upsertion task (sets status back to pending and pushes to queue).

---

## `PUT /embeddings/upsertion-tasks/cancel?task_id=...` — cancel_upsertion_task

### Description
Cancels an upsertion task by aborting the background job and marking it canceled.

---