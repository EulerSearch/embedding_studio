from embedding_studio.api.api_v1.schemas.upsert import (
    UpsertionTaskCreateRequest,
)
from embedding_studio.data_access.mongo.crud_base import CRUDBase
from embedding_studio.models.upsert import UpsertionTask, UpsertionTaskInDb


class CRUDUpsertion(
    CRUDBase[
        UpsertionTaskInDb,
        UpsertionTaskCreateRequest,
        UpsertionTask,
    ]
): ...
