from embedding_studio.models.items_handler import (
    BaseDataHandlingTask,
    BaseDataHandlingTaskCreateSchema,
)
from embedding_studio.models.task import (
    BaseModelOperationTask,
    BaseTaskCreateSchema,
    BaseTaskInDb,
)


class UpsertionTaskCreateSchema(
    BaseDataHandlingTaskCreateSchema, BaseTaskCreateSchema
):
    """
    A blueprint for creating tasks that add or update data in the system.
    "Upsert" is a combination of "update" and "insert" - so this handles
    both adding new data and updating existing data.
    """

    ...


class UpsertionTask(BaseDataHandlingTask, BaseModelOperationTask):
    """
    Handles the actual work of adding or updating vector embeddings in the
    database. It processes items and makes sure they're properly stored
    with their vector representations.
    """

    ...


class UpsertionTaskInDb(UpsertionTask, BaseTaskInDb):
    """
    The database-friendly version of an upsertion task. It includes
    everything needed to save the task details in the database for
    tracking and retrieval.
    """

    ...
