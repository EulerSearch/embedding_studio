import enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ItemProcessingFailureStage(str, enum.Enum):
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
        default="Unknown",
        description="Detailed error message explaining the failure",
    )
    failure_stage: ItemProcessingFailureStage = (
        ItemProcessingFailureStage.other
    )


class BaseDataHandlingTaskCreateSchema(BaseModel):
    items: List[DataItem] = Field(...)


class BaseDataHandlingTask(BaseModel):
    items: List[DataItem] = Field(default_factory=list)
    failed_items: List[FailedDataItem] = Field(default_factory=list)
