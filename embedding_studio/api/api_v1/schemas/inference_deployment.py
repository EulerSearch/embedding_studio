from typing import Any, Dict, Optional

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

from embedding_studio.db.common import PyObjectId
from embedding_studio.models.inference_deployment import (
    DeploymentStage,
    DeploymentStatus,
)


# Model for the deployment endpoints
class GreenDeploymentAfterFineTuningRequest(BaseModel):
    fine_tuning_task_id: str = Field(
        ..., description="ID of the task to deploy"
    )


class GreenDeploymentRequest(BaseModel):
    fine_tuning_method: str = Field(..., description="Plugin name to deploy")


class DeploymentRequest(BaseModel):
    task_id: str = Field(..., description="ID of the task to deploy")


class DeploymentInfo(BaseModel):
    fine_tuning_method: str = Field(..., description="Plugin name to deploy")
    stage: str = Field(..., description="Deployment stage to handle")


class DeploymentBaseResponse(BaseModel):
    fine_tuning_method: str = Field(...)
    stage: DeploymentStage = Field(...)
    created_at: AwareDatetime = Field(...)
    updated_at: AwareDatetime = Field(...)
    metadata: Optional[Dict[str, Any]] = None

    model_config = ConfigDict(populate_by_name=True)


class DeploymentResponse(DeploymentBaseResponse):
    id: PyObjectId = Field(alias="_id")
    status: DeploymentStatus = Field(...)
