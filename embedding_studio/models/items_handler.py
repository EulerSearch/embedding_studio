import enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ItemProcessingFailureStage(str, enum.Enum):
    """
    A list of specific points where processing an item might fail.
    Like a checklist of steps where things could go wrong - during download,
    while running the model, when splitting the data, etc. This helps
    pinpoint exactly where problems happen.
    """

    on_downloading = "on_downloading"
    on_inference = "on_inference"
    on_splitting = "on_splitting"
    on_upsert = "on_upsert"
    other = "other"


class DataItem(BaseModel):
    """
    Represents a single piece of data that will be processed by the system.
    It includes an ID to track the item, the actual data payload, and any
    extra information that might be helpful when processing it.
    """

    object_id: str
    payload: Optional[Dict[str, Any]] = None
    item_info: Optional[Dict[str, Any]] = None


class FailedDataItem(DataItem):
    """
    An extended version of DataItem that includes details about why it failed.
    When something goes wrong with processing an item, this captures both
    the item itself and information about what went wrong and at which stage.
    """

    detail: str = Field(
        default="Unknown",
        description="Detailed error message explaining the failure",
    )
    failure_stage: ItemProcessingFailureStage = (
        ItemProcessingFailureStage.other
    )


class BaseDataHandlingTaskCreateSchema(BaseModel):
    """
    A blueprint for creating tasks that process multiple data items at once.
    It's basically a container for a list of items that need to be processed
    together as a batch.
    """

    items: List[DataItem] = Field(...)


class BaseDataHandlingTask(BaseModel):
    """
    Keeps track of both successful and failed items during processing.
    This model helps manage batches of data items, maintaining separate lists
    for those that were processed successfully and those that failed.
    """

    items: List[DataItem] = Field(default_factory=list)
    failed_items: List[FailedDataItem] = Field(default_factory=list)
