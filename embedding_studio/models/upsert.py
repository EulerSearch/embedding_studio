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
    ...


class UpsertionTask(BaseDataHandlingTask, BaseModelOperationTask):
    ...


class UpsertionTaskInDb(UpsertionTask, BaseTaskInDb):
    ...
