import enum
from typing import Any, Dict, List, Optional

from pydantic import AwareDatetime, BaseModel, Field

from embedding_studio.db.common import PyObjectId


class UpsertionStatus(str, enum.Enum):
    pending = "pending"
    processing = "processing"
    done = "done"
    canceled = "canceled"
    error = "error"


class UpsertionFailureStage(str, enum.Enum):
    on_downloading = "on_downloading"
    on_inference = "on_inference"
    on_splitting = "on_splitting"
    on_upsert = "on_upsert"
    other = "other"


class DataItem(BaseModel):
    object_id: str
    payload: Optional[Dict[str, Any]] = None
    item_info: Optional[Dict[str, Any]] = None


class FailedDataItem(DataItem):
    detail: str = Field(
        ..., description="Detailed error message explaining the failure"
    )
    failure_stage: UpsertionFailureStage = UpsertionFailureStage.other


class UpsertionTaskCreateRequest(BaseModel):
    embedding_model_id: str = Field(
        description="Deployed embedding embedding_model ID"
    )
    fine_tuning_method: str = Field(
        description="Plugin name, which embedding_model to deploy"
    )
    items: List[DataItem] = Field(...)


class UpsertionTaskBaseResponse(BaseModel):
    fine_tuning_method: str = Field(...)
    embedding_model_id: str = Field(...)
    created_at: AwareDatetime = Field(...)
    updated_at: AwareDatetime = Field(...)


class UpsertionTaskResponse(UpsertionTaskBaseResponse):
    id: PyObjectId = Field(...)
    status: UpsertionStatus = Field(...)
    failed_items: List[FailedDataItem] = Field(...)


class UpsertionTaskDeleteResponse(UpsertionTaskBaseResponse):
    pass


class UpsertionTaskCancelResponse(UpsertionTaskBaseResponse):
    id: PyObjectId = Field(alias="_id")
    status: UpsertionStatus = Field(...)
