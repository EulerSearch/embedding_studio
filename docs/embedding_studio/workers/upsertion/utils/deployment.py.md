# Merged Documentation

## Documentation for `initiate_model_deployment_and_wait`

### Functionality

This function initiates model deployment and waits until the model is ready. It creates a deployment task, sends it asynchronously, and monitors its status. It handles pending, processing, failure, and timeout cases.

### Parameters

- `plugin`: A FineTuningMethod instance. Used to retrieve the inference client and control deployment.
- `task`: A ReindexTaskInDb instance. Links the deployment to a reindex task.
- `embedding_model`: A ModelParams instance. Contains model details like embedding_model_id.
- `deployment_worker`: A dramatiq Actor instance that sends the deployment task.

### Usage

This function deploys a model asynchronously and waits until it is ready for inference. It ensures proper error handling and timeout management.

#### Example

```python
deployment = initiate_model_deployment_and_wait(
    plugin,
    task,
    embedding_model,
    deployment_worker
)
# Proceed with using the model once it is ready.
```

---

## Documentation for `initiate_model_deletion`

### Functionality

This function initiates the deletion of a model. It creates a model deletion task using a given model identifier and sends the task to the designated deletion worker. The function logs the operation and raises exceptions upon failure.

### Parameters

- `task`: A ReindexTaskInDb instance that tracks the deletion process.
- `embedding_model_id`: A string identifier for the model to delete.
- `deletion_worker`: An Actor responsible for processing deletion tasks.

### Usage

- Purpose: To trigger the model deletion process by creating and sending a deletion task.

#### Example

```python
initiate_model_deletion(task, "model123", deletion_worker)
```

---

## Documentation for `blue_switch`

### Functionality

Handles the blue switch process by promoting the destination model to blue status and deleting the current blue model's collection. It ensures the destination model is ready for inference while cleaning up resources related to the previous blue model.

### Parameters

- `task` (ReindexTaskInDb): Contains information about the source and destination models. It is used to retrieve model iterations and manage the switch process.
- `deletion_worker` (Actor): Worker actor that initiates deletion of the source model's collection.

### Usage

- Purpose: Transition model roles by setting the destination model as blue and removing the source model's collections.

#### Example

```python
from embedding_studio.workers.upsertion.utils.deployment import blue_switch

# Assume task and deletion_worker are properly initialized
blue_switch(task, deletion_worker)
```