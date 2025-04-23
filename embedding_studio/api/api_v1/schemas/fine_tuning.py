from typing import Any, Dict, Optional

from embedding_studio.api.api_v1.internal_schemas.task import (
    BaseInternalTaskRequest,
    BaseInternalTaskResponse,
)


class FineTuningTaskRunRequest(BaseInternalTaskRequest):
    """
    Request schema for initiating a new embedding model fine-tuning operation.
    Supports idempotent execution through optional idempotency key.
    Includes deployment options to control how the fine-tuned model is applied.
    Enables batch-based training with optional metadata for tracking purposes.
    Critical for improving embedding quality through supervised learning.

    :param batch_id: Identifier for the training data batch to use for fine-tuning
    :param metadata: Additional information to store with the fine-tuning task
    :param idempotency_key: Unique key to prevent duplicate task creation
    :param deploy_as_blue: Whether to automatically deploy the model as blue/active
    :param wait_on_conflict: Whether to wait if there's a deployment conflict
    """

    batch_id: Optional[str] = None
    metadata: Optional[Dict] = None
    idempotency_key: Optional[str] = None

    deploy_as_blue: Optional[bool] = None
    wait_on_conflict: Optional[bool] = None


class FineTuningTaskResponse(BaseInternalTaskResponse):
    """
    Response schema reporting the status and results of a fine-tuning operation.
    Provides references to the produced model assets and their locations.
    Includes deployment status information for operational tracking.
    Maintains metadata continuity with the originating request.
    Essential for monitoring the model improvement lifecycle.

    :param batch_id: Identifier of the training data batch used for fine-tuning
    :param best_model_url: Location URL where the fine-tuned model is stored
    :param best_model_id: Unique identifier for the best performing model version
    :param metadata: Additional information stored with the fine-tuning task
    :param idempotency_key: Original key used to prevent duplicate task creation
    :param deploy_as_blue: Whether the model was deployed as blue/active
    :param wait_on_conflict: Whether the task waited on deployment conflicts
    """

    batch_id: Optional[str] = None
    best_model_url: Optional[str] = None
    best_model_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    idempotency_key: Optional[str] = None

    deploy_as_blue: Optional[bool] = None
    wait_on_conflict: Optional[bool] = None
