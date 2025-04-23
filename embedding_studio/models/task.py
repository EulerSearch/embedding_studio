import enum
from typing import Optional

from bson import ObjectId
from pydantic import AwareDatetime, BaseModel, Field, constr

from embedding_studio.db.common import PyObjectId
from embedding_studio.utils.datetime_utils import current_time


class ModelParams(BaseModel):
    """
    A simple container that holds the ID of an embedding model.
    Think of it as a label or reference to a specific model that
    your system needs to work with.
    """

    embedding_model_id: str = Field(...)


class BaseTaskMetadata(BaseModel):
    """
    Holds basic information about a task, like its ID and which task created it
    (if any). This helps keep track of how tasks are connected to each other,
    like a family tree of tasks.
    """

    id: Optional[PyObjectId] = Field(default=ObjectId, alias="_id")

    parent_id: Optional[PyObjectId] = Field(default=None)


class BaseTaskCreateSchema(ModelParams, BaseTaskMetadata):
    """
    A blueprint for creating new tasks. It combines information about
    which model to use and the task's basic details. It's like a form you
    fill out when you want to start a new task.
    """

    ...


class TaskStatus(str, enum.Enum):
    """
    A list of possible states a task can be in, like "pending", "processing",
    or "done". This helps everyone understand where a task is in its lifecycle,
    like tracking a package delivery status.
    """

    pending = "pending"
    processing = "processing"
    done = "done"
    canceled = "canceled"
    failed = "failed"
    refused = "refused"


class BaseTaskInDb(BaseModel):
    """
    A version of a task that's ready to be stored in the database.
    It adds database-specific details like the broker ID, which helps
    the system know which worker should handle the task.
    """

    id: Optional[PyObjectId] = Field(default=ObjectId, alias="_id")
    broker_id: Optional[str] = Field(default=None)

    parent_id: Optional[PyObjectId] = Field(default=None)


class BaseTaskInfo(BaseModel):
    """
    Contains all the details about a task's current status, when it was
    created or updated, and any parent tasks. This is the model that
    shows users how their task is progressing.
    """

    status: TaskStatus = Field(default=TaskStatus.pending)
    detail: Optional[constr(max_length=1500)] = Field(
        None,
        description="Details or information about the current status of the task",
    )

    created_at: AwareDatetime = Field(default_factory=current_time)
    updated_at: AwareDatetime = Field(default_factory=current_time)

    parent_id: Optional[PyObjectId] = Field(default=None)


class BaseModelOperationTask(BaseTaskInfo, ModelParams):
    """
    A special type of task that works directly with embedding models.
    It combines task status information with model details, making it
    easy to perform operations on specific models.
    """

    ...
