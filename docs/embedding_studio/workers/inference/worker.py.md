# Merged Documentation

## Documentation for `model_deployment_worker`

### Functionality

Handles the model deployment process in the inference worker. It receives a task identifier and calls the deployment utility to initiate model deployment procedures.

### Parameters

- `task_id`: A unique string identifier for the deployment task.

### Usage

- **Purpose** - Triggers model deployment workflows by invoking the underlying deployment handler.

#### Example

```python
from embedding_studio.workers.inference.worker import \
    model_deployment_worker

task_id = "example_deployment_task_id"
model_deployment_worker(task_id)
```

---

## Documentation for `model_deletion_worker`

### Functionality

The model_deletion_worker is a Dramatiq actor that handles deletion of an embedding model from the Triton Inference Server. It validates the deletion request, checks plugin support, and safely removes model directories.

### Parameters

- `task_id`: A string identifier for the deletion task to process.

### Usage

- **Purpose**: Processes deletion tasks by delegating to the handle_deletion function, ensuring validations and safe removals.

#### Example

When a deletion task is enqueued, call:

```python
model_deletion_worker("task_identifier")
```