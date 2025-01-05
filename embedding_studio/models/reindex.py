from typing import List, Optional

from bson import ObjectId
from pydantic import BaseModel, Field

from embedding_studio.db.common import PyObjectId
from embedding_studio.models.items_handler import (
    BaseDataHandlingTask,
    FailedDataItem,
)
from embedding_studio.models.task import (
    BaseTaskInDb,
    BaseTaskInfo,
    BaseTaskMetadata,
    ModelParams,
)


class ReindexTaskCreateSchema(BaseModel):
    id: Optional[PyObjectId] = Field(default=ObjectId, alias="_id")
    source: ModelParams = Field(...)
    dest: ModelParams = Field(...)

    deploy_as_blue: Optional[bool] = Field(...)
    wait_on_conflict: Optional[bool] = Field(...)

    parent_id: Optional[PyObjectId] = Field(default=None)


class ReindexSubtaskCreateSchema(BaseTaskInfo):
    limit: int = Field(...)
    offset: Optional[int] = Field(...)
    source: ModelParams = Field(...)
    dest: ModelParams = Field(...)


class ReindexTask(BaseTaskInfo):
    source: ModelParams = Field(...)
    dest: ModelParams = Field(...)

    progress: float = Field(
        default=0.0,
        ge=0.0,
        description="Task progress represented as a percentage (0-100)",
    )

    count: Optional[int] = Field(
        default=0, ge=0, description="Number of processed items so far"
    )
    total: Optional[int] = Field(
        default=None, ge=0.0, description="Total number of items to process"
    )

    children: List[PyObjectId] = Field(default_factory=list)
    failed_items: List[FailedDataItem] = Field(default_factory=list)

    deploy_as_blue: Optional[bool] = Field(...)
    wait_on_conflict: Optional[bool] = Field(...)

    def add_count(self, additional_count):
        self.count += additional_count
        self.progress = self.count / self.total


class ReindexSubtask(BaseDataHandlingTask, BaseTaskInfo, BaseTaskMetadata):
    limit: int = Field(...)
    offset: Optional[int] = Field(...)

    source: ModelParams = Field(...)
    dest: ModelParams = Field(...)


class ReindexTaskInDb(BaseTaskInDb, ReindexTask):
    ...


class ReindexSubtaskInDb(BaseTaskInDb, ReindexSubtask):
    ...
