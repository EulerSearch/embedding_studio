from typing import Optional

from bson import ObjectId
from pydantic import Field

from embedding_studio.db.common import PyObjectId
from embedding_studio.models.task import BaseTaskInDb, BaseTaskInfo


class SessionForImprovementCreateSchema(BaseTaskInfo):
    id: Optional[PyObjectId] = Field(default=ObjectId, alias="_id")
    session_id: str = Field(...)


class SessionForImprovement(BaseTaskInfo):
    session_id: str = Field(...)


class SessionForImprovementInDb(SessionForImprovement, BaseTaskInDb):
    ...
