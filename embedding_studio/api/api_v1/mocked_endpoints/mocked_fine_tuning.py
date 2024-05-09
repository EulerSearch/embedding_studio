import logging
from typing import Any

from fastapi import APIRouter

from embedding_studio.api.api_v1.schemas.fine_tuning import (
    FineTuningTaskCreateRequest,
    FineTuningTaskResponse,
)
from embedding_studio.context.app_context import context
from embedding_studio.workers.fine_tuning.mocked_worker import (
    fine_tuning_mocked_worker,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/task",
    response_model=FineTuningTaskResponse,
    response_model_by_alias=False,
    response_model_exclude_none=True,
)
def create_fine_tuning_task(
    body: FineTuningTaskCreateRequest,
) -> Any:
    """Simulate a creation a new fine-tuning task.

    :param body: Request body.
    :return: Created task details.
    """
    # TODO: check if task with the same batch_id and params already exists
    logger.debug(f"POST /task: {body}")
    if body.idempotency_key:
        task = context.fine_tuning_task.get_by_idempotency_key(
            body.idempotency_key
        )
        if task is not None:
            return task
    task = context.fine_tuning_task.create(schema=body, return_obj=True)
    message = fine_tuning_mocked_worker.send(str(task.id))
    logger.debug(f"fine_tuning_worker message: {message}")
    task.broker_id = message.message_id
    context.fine_tuning_task.update(obj=task)
    return task
