# Documentation for `create_fine_tuning_task`

## Functionality

Simulates creation of a new fine-tuning task. This function first retrieves the iteration details using the provided `embedding_model_id`. It then verifies plugin support and checks for an existing task using the `idempotency_key`. If no task is found, it creates a new one and dispatches it to the fine-tuning worker.

## Parameters

- `body`: A `FineTuningTaskRunRequest` object containing necessary fields such as `embedding_model_id` and optionally an `idempotency_key`.

## Usage

- **Purpose**: Initiate a fine-tuning process after validating the model iteration and plugin compatibility.

### Example

Request:
```python
body = FineTuningTaskRunRequest(
    embedding_model_id="model123",
    idempotency_key="unique_key_abc"
)
task = create_fine_tuning_task(body)
```