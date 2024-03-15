from typing import Any, Dict, Optional

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

from embedding_studio.db.common import PyObjectId
from embedding_studio.models.fine_tuning import FineTuningStatus


class FineTuningTaskBaseResponse(BaseModel):
    fine_tuning_method: str = Field(...)
    created_at: AwareDatetime = Field(...)
    updated_at: AwareDatetime = Field(...)
    batch_id: Optional[str] = None
    best_model_url: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    idempotency_key: Optional[str] = None

    model_config = ConfigDict(populate_by_name=True)


class FineTuningTaskResponse(FineTuningTaskBaseResponse):
    id: PyObjectId = Field(alias="_id")
    status: FineTuningStatus = Field(...)


class FineTuningTaskCreateRequest(BaseModel):
    fine_tuning_method: str = Field(...)
    batch_id: Optional[str] = None
    metadata: Optional[Dict] = None
    idempotency_key: Optional[str] = None


class FineTuningTaskDeleteResponse(FineTuningTaskBaseResponse):
    pass


class FineTuningTaskCancelResponse(FineTuningTaskBaseResponse):
    id: PyObjectId = Field(alias="_id")
    status: FineTuningStatus = Field(...)
