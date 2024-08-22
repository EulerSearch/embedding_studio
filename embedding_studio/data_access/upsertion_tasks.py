from embedding_studio.data_access.mongo.crud_base import CRUDBase
from embedding_studio.models.upsert import (
    UpsertionTask,
    UpsertionTaskCreateSchema,
    UpsertionTaskInDb,
)


class CRUDUpsertion(
    CRUDBase[
        UpsertionTaskInDb,
        UpsertionTaskCreateSchema,
        UpsertionTask,
    ]
):
    ...
