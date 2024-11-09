from embedding_studio.data_access.model_transfer_tasks import (
    CRUDModelTransferTasks,
)
from embedding_studio.models.reindex_lock import (
    ReindexLock,
    ReindexLockCreateSchema,
    ReindexLockInDb,
)


class CRUDReindexLocks(
    CRUDModelTransferTasks[
        ReindexLockInDb,
        ReindexLockCreateSchema,
        ReindexLock,
    ]
):
    ...
