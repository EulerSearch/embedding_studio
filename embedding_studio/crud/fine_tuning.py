from embedding_studio.crud.base import CRUDBase
from embedding_studio.db import mongo
from embedding_studio.schemas.fine_tuning import (
    FineTuningTask,
    FineTuningTaskCreate,
    FineTuningTaskInDb,
)


class CRUDFineTuning(
    CRUDBase[FineTuningTaskInDb, FineTuningTaskCreate, FineTuningTask]
):
    ...


fine_tuning_task = CRUDFineTuning(
    mongo.finetuning_mongo_database["fine_tuning"], FineTuningTaskInDb
)
