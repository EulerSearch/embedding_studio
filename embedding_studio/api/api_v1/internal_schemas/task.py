from pydantic import Field

from embedding_studio.api.api_v1.schemas.task import (
    BaseTaskRequest,
    BaseTaskResponse,
)


class BaseInternalTaskRequest(BaseTaskRequest):
    embedding_model_id: str = Field(
        description="Deployed embedding embedding_model ID"
    )


class BaseInternalTaskResponse(BaseTaskResponse):
    embedding_model_id: str = Field(...)
