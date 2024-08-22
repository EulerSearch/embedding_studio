from embedding_studio.data_access.mongo.crud_base import CRUDBase
from embedding_studio.models.delete import (
    DeletionTask,
    DeletionTaskCreateSchema,
    DeletionTaskInDb,
)


class CRUDDeletion(
    CRUDBase[
        DeletionTask,
        DeletionTaskCreateSchema,
        DeletionTaskInDb,
    ]
):
    ...
