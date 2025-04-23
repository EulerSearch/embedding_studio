from typing import List

from pydantic import BaseModel, Field

from embedding_studio.models.task import (
    BaseModelOperationTask,
    BaseTaskCreateSchema,
    BaseTaskInDb,
)


class FailedItemIdWithDetail(BaseModel):
    """
    Keeps track of items that couldn't be deleted, along with the reason why.
    It's like an error report that helps you understand what went wrong
    with specific items during deletion.
    """

    object_id: str
    detail: str = Field(
        description="Detailed error message explaining the failure"
    )


class DeletionTaskCreateSchema(BaseTaskCreateSchema):
    """
    A blueprint for creating tasks that remove data from the system.
    It specifies which objects to delete, identified by their IDs.
    """

    object_ids: List[str] = Field(...)


class DeletionTask(BaseModelOperationTask):
    """
    Handles the work of removing vector embeddings from the database.
    It keeps track of both the items to delete and any items that
    couldn't be deleted (along with why they failed).
    """

    object_ids: List[str] = Field(...)
    failed_item_ids: List[FailedItemIdWithDetail] = Field(default_factory=list)


class DeletionTaskInDb(DeletionTask, BaseTaskInDb):
    """
    The database-friendly version of a deletion task. It includes
    everything needed to save the deletion job's details in the database.
    """

    ...
