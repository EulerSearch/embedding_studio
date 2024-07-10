from typing import Any, Dict, Optional

from pydantic import AwareDatetime, BaseModel, Field

from embedding_studio.db.common import PyObjectId
from embedding_studio.models.inference_deployment_tasks import (
    ModelDeploymentStatus,
)


class ModelDeploymentRequest(BaseModel):
    model_id: str = Field(description="Model to deploy")
    fine_tuning_method: str = Field(
        description="Plugin name, which model to deploy"
    )


class ModelDeploymentBaseResponse(BaseModel):
    fine_tuning_method: str = Field(...)
    embedding_model_id: str = Field(...)
    created_at: AwareDatetime = Field(...)
    updated_at: AwareDatetime = Field(...)
    metadata: Optional[Dict[str, Any]] = None


class ModelDeploymentResponse(ModelDeploymentBaseResponse):
    id: PyObjectId = Field(alias="_id")
    status: ModelDeploymentStatus = Field(...)


class ModelDeletionRequest(BaseModel):
    embedding_model_id: str = Field(description="Model to delete")
    fine_tuning_method: str = Field(
        description="Plugin name, which model to deploy"
    )


class ModelDeletionBaseResponse(BaseModel):
    fine_tuning_method: str = Field(...)
    embedding_model_id: str = Field(...)
    created_at: AwareDatetime = Field(...)
    updated_at: AwareDatetime = Field(...)
    metadata: Optional[Dict[str, Any]] = None


class ModelDeletionResponse(ModelDeletionBaseResponse):
    id: PyObjectId = Field(alias="_id")
    status: ModelDeploymentStatus = Field(...)
