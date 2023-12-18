from embedding_studio.api.api_v1.schemas.fine_tuning import (
    FineTuningTaskCreate,
)
from embedding_studio.data_access.mongo.crud_base import CRUDBase
from embedding_studio.models.fine_tuning import (
    FineTuningTask,
    FineTuningTaskInDb,
)


class CRUDFineTuning(
    CRUDBase[FineTuningTaskInDb, FineTuningTaskCreate, FineTuningTask]
):
    ...
