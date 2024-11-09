from embedding_studio.data_access.mongo.crud_base import CRUDBase
from embedding_studio.models.reindex import (
    ReindexSubtask,
    ReindexSubtaskCreateSchema,
    ReindexSubtaskInDb,
    ReindexTask,
    ReindexTaskCreateSchema,
    ReindexTaskInDb,
)


class CRUDReindexTasks(
    CRUDBase[
        ReindexTask,
        ReindexTaskCreateSchema,
        ReindexTaskInDb,
    ]
):
    ...


class CRUDReindexSubtasks(
    CRUDBase[
        ReindexSubtask,
        ReindexSubtaskCreateSchema,
        ReindexSubtaskInDb,
    ]
):
    ...
