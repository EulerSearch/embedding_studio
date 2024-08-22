from pydantic import Field

from embedding_studio.api.api_v1.schemas.task import (
    BaseTaskRequest,
    BaseTaskResponse,
)


class BaseInternalTaskRequest(BaseTaskRequest):
    embedding_model_id: str = Field(
        description="Deployed embedding embedding_model ID"
    )
    fine_tuning_method: str = Field(
        description="Plugin name, which embedding_model to deploy"
    )


class BaseInternalTaskResponse(BaseTaskResponse):
    fine_tuning_method: str = Field(...)
    embedding_model_id: str = Field(...)
