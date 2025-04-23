# Merged Documentation

## Upsertion Task Methods

### `create_upsertion_task`

#### Functionality
This endpoint creates a new upsertion task for embedding creation. If a task with the provided `task_id` already exists, it returns the existing task. Otherwise, it retrieves embedding model details from the vector database, creates a new task, and dispatches it using the task worker.

#### Parameters
- **body**: `UpsertionTaskRunRequest` object containing:
  - `task_id` (optional): Identifier for the task.
  - `items`: List of items for which embeddings must be generated or updated.

#### Usage
- **Purpose**: To create and trigger an upsertion task for embedding operations.

##### Example
Request body:
```json
{
  "task_id": "12345",
  "items": [/* item details */]
}
```
Response (on success):
```json
{
  "task_id": "12345",
  "status": "created",
  /* additional task details */
}
```

### `create_categories_upsertion_task`

#### Functionality
Creates a new categories upsertion task by checking if a task with the provided ID exists. If not, it prepares a new task using the categories vectordb collection and sends it for processing.

#### Parameters
- **body**: `UpsertionTaskRunRequest` object containing task details, including an optional `task_id` and a list of `items` to upsert.

#### Usage
- **Purpose**: Triggers processing of a categories upsertion task.

##### Example
A POST request to `/embeddings/upsertion-tasks/categories/run` with a JSON body such as:
```json
{
  "task_id": "optional-id",
  "items": [ ... ]
}
```