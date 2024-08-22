from typing import List

from pydantic import BaseModel, Field

from embedding_studio.models.task import (
    BaseModelOperationTask,
    BaseTaskCreateSchema,
    BaseTaskInDb,
)


class FailedItemIdWithDetail(BaseModel):
    object_id: str
    detail: str = Field(
        description="Detailed error message explaining the failure"
    )


class DeletionTaskCreateSchema(BaseTaskCreateSchema):
    object_ids: List[str] = Field(...)


class DeletionTask(BaseModelOperationTask):
    object_ids: List[str] = Field(...)
    failed_item_ids: List[FailedItemIdWithDetail] = Field(default_factory=list)


class DeletionTaskInDb(DeletionTask, BaseTaskInDb):
    ...
