from typing import Any, Dict, Optional

from embedding_studio.api.api_v1.internal_schemas.task import (
    BaseInternalTaskRequest,
    BaseInternalTaskResponse,
)


class FineTuningTaskRunRequest(BaseInternalTaskRequest):
    batch_id: Optional[str] = None
    metadata: Optional[Dict] = None
    idempotency_key: Optional[str] = None

    deploy_as_blue: Optional[bool] = None
    wait_on_conflict: Optional[bool] = None


class FineTuningTaskResponse(BaseInternalTaskResponse):
    batch_id: Optional[str] = None
    best_model_url: Optional[str] = None
    best_model_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    idempotency_key: Optional[str] = None

    deploy_as_blue: Optional[bool] = None
    wait_on_conflict: Optional[bool] = None
