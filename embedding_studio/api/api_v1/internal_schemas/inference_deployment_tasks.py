from typing import Any, Dict, Optional

from embedding_studio.api.api_v1.internal_schemas.task import (
    BaseInternalTaskRequest,
    BaseInternalTaskResponse,
)


class ModelDeploymentRequest(BaseInternalTaskRequest):
    ...


class ModelDeploymentResponse(BaseInternalTaskResponse):
    metadata: Optional[Dict[str, Any]] = None


class ModelDeletionRequest(BaseInternalTaskRequest):
    ...


class ModelDeletionResponse(BaseInternalTaskResponse):
    metadata: Optional[Dict[str, Any]] = None
