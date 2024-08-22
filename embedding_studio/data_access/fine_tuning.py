from embedding_studio.data_access.mongo.crud_base import CRUDBase
from embedding_studio.models.fine_tuning import (
    FineTuningTask,
    FineTuningTaskCreateSchema,
    FineTuningTaskInDb,
)


class CRUDFineTuning(
    CRUDBase[FineTuningTaskInDb, FineTuningTaskCreateSchema, FineTuningTask]
):
    ...
