from typing import Optional

from bson import ObjectId
from pydantic import Field

from embedding_studio.db.common import PyObjectId
from embedding_studio.models.task import BaseTaskInDb, BaseTaskInfo


class SessionForImprovementCreateSchema(BaseTaskInfo):
    """
    A blueprint for creating tasks that will process user sessions to improve
    search results. This tracks specific user interaction sessions that contain
    valuable feedback (like clicks) which can be used to make the system smarter.
    """

    id: Optional[PyObjectId] = Field(default=ObjectId, alias="_id")
    session_id: str = Field(...)


class SessionForImprovement(BaseTaskInfo):
    """
    Represents a user session that contains feedback data worth analyzing.
    This model tracks sessions where users interacted with search results,
    providing implicit feedback that can be used to improve future results.
    """

    session_id: str = Field(...)


class SessionForImprovementInDb(SessionForImprovement, BaseTaskInDb):
    """
    The database-friendly version of a session improvement task. It stores
    information about which user sessions should be analyzed to improve
    the system's understanding of what results are most relevant.
    """

    ...
