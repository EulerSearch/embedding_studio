# Documentation for Inference Deployment Tasks API

---

## `POST /internal/inference-deployment/deploy` — deploy

### Description
Schedules deployment of an embedding model to the inference system using green deployment strategy. This endpoint ensures the model is configured for serving and logs deployment lifecycle metadata.

### Request Fields
- `embedding_model_id` *(str)*: The ID of the model to deploy.

### Request Example
```json
{
  "embedding_model_id": "embed-deploy-123"
}
```

### Response Example
```json
{
  "id": "deploy-42",
  "embedding_model_id": "embed-deploy-123",
  "status": "pending",
  "created_at": "2024-05-21T15:00:00Z",
  "updated_at": "2024-05-21T15:00:00Z",
  "metadata": {
    "deployment_target": "inference-cluster-prod",
    "initiated_by": "admin"
  }
}
```

---

## `GET /internal/inference-deployment/deploy/{task_id}` — get_deploy_task

### Description
Returns the current status and metadata of a deployment task. Can be used to monitor progress and trace failures during rollout.

---

## `GET /internal/inference-deployment/deploy-status/{embedding_model_id}` — get_model_deploy_status

### Description
Fetches the latest deployment task status for a given model. Useful for polling deployment lifecycle externally.

---

## `POST /internal/inference-deployment/delete` — delete

### Description
Schedules a model for removal from the inference system.

### Request Fields
- `embedding_model_id` *(str)*: ID of the model to delete from inference.

### Request Example
```json
{
  "embedding_model_id": "embed-deploy-123"
}
```

### Response Example
```json
{
  "id": "delete-42",
  "embedding_model_id": "embed-deploy-123",
  "status": "pending",
  "created_at": "2024-05-21T15:10:00Z",
  "updated_at": "2024-05-21T15:10:00Z",
  "metadata": {
    "deletion_type": "manual",
    "reason": "model deprecated"
  }
}
```

---

## `GET /internal/inference-deployment/delete/{task_id}` — get_delete_task

### Description
Fetches metadata and lifecycle state of a specific model deletion request.

---

## `GET /internal/inference-deployment/delete-status/{embedding_model_id}` — get_model_delete_status

### Description
Returns the most recent deletion task info for the specified model. Helps ensure that cleanup was initiated or completed as expected.

---