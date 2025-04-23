from typing import Any, Dict, Optional

from embedding_studio.api.api_v1.internal_schemas.task import (
    BaseInternalTaskRequest,
    BaseInternalTaskResponse,
)


class ModelDeploymentRequest(BaseInternalTaskRequest):
    """
    Request schema for deploying embedding models to the inference service.
    Inherits model targeting from the base internal task structure.
    Enables controlled introduction of new models into the production environment.
    """

    ...


class ModelDeploymentResponse(BaseInternalTaskResponse):
    """
    Response structure tracking embedding model deployment status and outcomes.
    Includes optional metadata about the deployment process and resulting service.
    Provides the complete lifecycle context of the model deployment operation.
    Enables clients to monitor deployment progress and confirm successful activation.
    """

    metadata: Optional[Dict[str, Any]] = None


class ModelDeletionRequest(BaseInternalTaskRequest):
    """
    Request schema for removing embedding models from the inference service.
    Specifies which model should be decommissioned from the production environment.
    Enables proper resource cleanup when models are no longer needed.
    Helps maintain system efficiency by eliminating unused model resources.
    """

    ...


class ModelDeletionResponse(BaseInternalTaskResponse):
    """
    Response structure documenting embedding model deletion results.
    Contains optional metadata about the removal process and cleanup operations.
    Tracks the complete lifecycle of the model deletion task.
    Provides confirmation that model resources have been properly released.
    """

    metadata: Optional[Dict[str, Any]] = None
