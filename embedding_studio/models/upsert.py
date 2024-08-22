import enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from embedding_studio.models.task import (
    BaseModelOperationTask,
    BaseTaskCreateSchema,
    BaseTaskInDb,
)


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


class UpsertionTaskCreateSchema(BaseTaskCreateSchema):
    items: List[DataItem] = Field(...)


class FailedDataItem(DataItem):
    detail: str = Field(
        description="Detailed error message explaining the failure"
    )
    failure_stage: UpsertionFailureStage = UpsertionFailureStage.other


class UpsertionTask(BaseModelOperationTask):
    items: List[DataItem] = Field(...)
    failed_items: List[FailedDataItem] = Field(default_factory=list)


class UpsertionTaskInDb(UpsertionTask, BaseTaskInDb):
    ...
