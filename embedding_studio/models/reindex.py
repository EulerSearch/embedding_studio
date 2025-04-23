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
    """
    A blueprint for tasks that move data between different embedding models.
    Think of it like a migration plan - it defines how to transfer data
    from one model to another, with options for how to handle the switch.
    """

    id: Optional[PyObjectId] = Field(default=ObjectId, alias="_id")
    source: ModelParams = Field(...)
    dest: ModelParams = Field(...)

    deploy_as_blue: Optional[bool] = Field(...)
    wait_on_conflict: Optional[bool] = Field(...)

    parent_id: Optional[PyObjectId] = Field(default=None)


class ReindexSubtaskCreateSchema(BaseTaskInfo):
    """
    A blueprint for smaller chunks of work within a reindexing job.
    Since reindexing can involve lots of data, this breaks it into
    manageable pieces that can be processed separately.
    """

    limit: int = Field(...)
    offset: Optional[int] = Field(...)
    source: ModelParams = Field(...)
    dest: ModelParams = Field(...)


class ReindexTask(BaseTaskInfo):
    """
    Manages the overall process of transferring data between embedding models.
    It tracks progress, coordinates all the subtasks, and handles the final
    deployment of the new model when everything is done.
    """

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

    deploy_as_blue: Optional[bool] = Field(...)
    wait_on_conflict: Optional[bool] = Field(...)

    def add_count(self, additional_count):
        self.count += additional_count
        self.progress = self.count / self.total


class ReindexSubtask(BaseDataHandlingTask, BaseTaskInfo, BaseTaskMetadata):
    """
    Handles one batch of data during reindexing. Each subtask processes
    a specific chunk of data (like items 1-100), making it possible to
    work on different parts of the data at the same time.
    """

    limit: int = Field(...)
    offset: Optional[int] = Field(...)

    source: ModelParams = Field(...)
    dest: ModelParams = Field(...)


class ReindexTaskInDb(BaseTaskInDb, ReindexTask):
    """
    The database-friendly version of a reindex task. It includes
    everything needed to save the main reindexing job's details
    in the database.
    """

    ...


class ReindexSubtaskInDb(BaseTaskInDb, ReindexSubtask):
    """
    The database-friendly version of a reindex subtask. It stores
    information about one batch of the reindexing process in the database.
    """

    ...
