from typing import Optional

from bson import ObjectId
from pydantic import Field

from embedding_studio.db.common import PyObjectId
from embedding_studio.models.task import (
    BaseModelOperationTask,
    BaseTaskCreateSchema,
    BaseTaskInDb,
)


class ReindexLockCreateSchema(BaseTaskCreateSchema):
    dst_embedding_model_id: str = Field(...)


class ReindexLock(BaseModelOperationTask):
    dst_embedding_model_id: str = Field(...)

    parent_id: Optional[PyObjectId] = Field(default=ObjectId, alias="_id")


class ReindexLockInDb(ReindexLock, BaseTaskInDb):
    ...
