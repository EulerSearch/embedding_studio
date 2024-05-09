import enum
from typing import Any, Dict, Optional

from bson import ObjectId
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

from embedding_studio.db.common import PyObjectId
from embedding_studio.utils.datetime_utils import current_time


class DeploymentStatus(str, enum.Enum):
    pending = "pending"
    processing = "processing"
    done = "done"
    canceled = "canceled"
    error = "error"


class DeploymentStage(str, enum.Enum):
    green = "green"
    blue = "blue"
    revert = "revert"


class DeploymentTask(BaseModel):
    fine_tuning_method: str = Field(...)
    status: DeploymentStatus = Field(default=DeploymentStatus.pending)
    stage: DeploymentStage = Field(default=DeploymentStage.green)
    created_at: AwareDatetime = Field(default_factory=current_time)
    updated_at: AwareDatetime = Field(default_factory=current_time)
    metadata: Optional[Dict[str, Any]] = None
    idempotency_key: Optional[str] = None

    model_config = ConfigDict(populate_by_name=True)


class DeploymentTaskInDb(DeploymentTask):
    id: Optional[PyObjectId] = Field(default=ObjectId, alias="_id")
    broker_id: Optional[str] = Field(default=None)
