from typing import Any, Dict, Optional

from pydantic import ConfigDict

from embedding_studio.models.task import (
    BaseModelOperationTask,
    BaseTaskCreateSchema,
    BaseTaskInDb,
)


class FineTuningTaskCreateSchema(BaseTaskCreateSchema):
    """
    A blueprint for creating tasks that improve embedding models using real user data.
    This schema defines how to set up a fine-tuning job, including which data batch
    to use and deployment preferences once the model is improved.
    """

    batch_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    idempotency_key: Optional[str] = None

    deploy_as_blue: Optional[bool] = None
    wait_on_conflict: Optional[bool] = None


class FineTuningTask(BaseModelOperationTask):
    """
    Represents a task that customizes embedding models based on user interactions.
    It tracks the entire fine-tuning process, from the initial model to the best
    improved version, with information about where to find the new model.
    """

    batch_id: Optional[str] = None
    best_run_id: Optional[str] = None
    best_model_url: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    idempotency_key: Optional[str] = None

    deploy_as_blue: Optional[bool] = None
    wait_on_conflict: Optional[bool] = None

    model_config = ConfigDict(populate_by_name=True)


class FineTuningTaskInDb(FineTuningTask, BaseTaskInDb):
    """
    The database-friendly version of a fine-tuning task. It stores all the
    information about the fine-tuning process, including the original model,
    the newly created model, and relevant metadata.
    """

    ...
