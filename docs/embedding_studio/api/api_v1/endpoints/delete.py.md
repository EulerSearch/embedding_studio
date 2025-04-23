# Documentation for Deletion Tasks

## create_deletion_task

### Functionality

Creates a new deletion task. It first checks whether a task with the given `task_id` exists. If it exists, the task details are returned. Otherwise, a new task is created from the given `object_ids` extracted from the request body. The task is then sent using a dedicated worker, and the response is returned. In case of failure, an HTTP 500 error is raised.

### Parameters

- **body (DeletionTaskRunRequest)**: Request body that includes an optional `task_id` and the list of `object_ids` to be deleted.

### Usage

- **Purpose**: Initialize a deletion task for embeddings in a specific collection.

#### Example

A simple usage example:

```python
request_body = {
    "task_id": "optional_task_id",
    "object_ids": ["id1", "id2"]
}
response = create_deletion_task(request_body)
```

---

## create_categories_deletion_task

### Functionality

Creates and dispatches a new deletion task for categories. It first checks if a task with the provided `task_id` already exists and returns it. If no such task exists, it retrieves the categories collection info, creates a new deletion task using these details, and dispatches it via a background worker.

### Parameters

- **body (DeletionTaskRunRequest)**: This object contains:
  - `task_id` (optional): Used to check for an existing task if provided.
  - `object_ids`: A list of IDs representing the categories to be deleted.

### Usage

- **Purpose**: To initiate a categories deletion task by sending a proper request payload to the endpoint.

#### Example

A sample request payload:

```json
{
    "task_id": "unique-task-id",
    "object_ids": ["cat1", "cat2"]
}
```

POST the payload to `/embeddings/deletion-tasks/categories/run`