from typing import Any, Dict, Optional

from pydantic import ConfigDict

from embedding_studio.models.task import (
    BaseModelOperationTask,
    BaseTaskCreateSchema,
    BaseTaskInDb,
)


class FineTuningTaskCreateSchema(BaseTaskCreateSchema):
    batch_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    idempotency_key: Optional[str] = None


class FineTuningTask(BaseModelOperationTask):
    batch_id: Optional[str] = None
    best_run_id: Optional[str] = None
    best_model_url: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    idempotency_key: Optional[str] = None

    model_config = ConfigDict(populate_by_name=True)


class FineTuningTaskInDb(FineTuningTask, BaseTaskInDb):
    ...
