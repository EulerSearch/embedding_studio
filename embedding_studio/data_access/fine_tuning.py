from embedding_studio.api.api_v1.schemas.fine_tuning import (
    FineTuningTaskCreateRequest,
)
from embedding_studio.data_access.mongo.crud_base import CRUDBase
from embedding_studio.models.fine_tuning import (
    FineTuningTask,
    FineTuningTaskInDb,
)


class CRUDFineTuning(
    CRUDBase[FineTuningTaskInDb, FineTuningTaskCreateRequest, FineTuningTask]
): ...
