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
    """
    A blueprint for creating a lock that prevents concurrent operations
    on models being reindexed. It specifies both the source model and
    the destination model involved in the reindexing process.
    """

    dst_embedding_model_id: str = Field(...)


class ReindexLock(BaseModelOperationTask):
    """
    Represents an active lock that protects models during reindexing.
    This prevents other operations from interfering with models that are
    in the middle of a reindexing process, like putting an "in use" sign
    on them.
    """

    dst_embedding_model_id: str = Field(...)
    parent_id: Optional[PyObjectId] = Field(default=ObjectId, alias="_id")


class ReindexLockInDb(ReindexLock, BaseTaskInDb):
    """
    The database-friendly version of a reindex lock. It includes everything
    needed to store the lock information in the database so the system knows
    which models are currently unavailable for other operations.
    """

    ...
