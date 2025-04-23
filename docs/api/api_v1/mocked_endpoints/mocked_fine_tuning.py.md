# Documentation for create_fine_tuning_task

## Functionality
The `create_fine_tuning_task` method simulates the creation of a fine-tuning task. It verifies the request, checks for duplicate tasks using an idempotency key, and validates that the plugin is basic. The task is then sent to a mocked worker, and details of the created task are returned.

## Parameters
- **body**: An instance of `FineTuningTaskRunRequest`.
  - **embedding_model_id**: Identifier to retrieve the task iteration.
  - **idempotency_key**: (Optional) Key to avoid creating duplicate tasks.

## Usage
- **Purpose**: To initiate a fine-tuning process for basic models only. (Tasks for non-basic plugins are rejected.)

### API Description
- **HTTP Method**: POST
- **Endpoint**: /task
- **Request**: `FineTuningTaskRunRequest`
- **Response**: `FineTuningTaskResponse`

### Example
```bash
curl -X POST "http://<host>/task" \
     -H "Content-Type: application/json" \
     -d '{"embedding_model_id": "model123", "idempotency_key": "key123"}'
```