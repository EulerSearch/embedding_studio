from embedding_studio.data_access.mongo.crud_base import CRUDBase
from embedding_studio.models.improvement import (
    SessionForImprovement,
    SessionForImprovementCreateSchema,
    SessionForImprovementInDb,
)


class CRUDSessionsForImprovement(
    CRUDBase[
        SessionForImprovementInDb,
        SessionForImprovementCreateSchema,
        SessionForImprovement,
    ]
):
    ...
