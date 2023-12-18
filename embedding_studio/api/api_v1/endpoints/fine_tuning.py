import logging
from typing import Any, List

from dramatiq_abort import abort as dramatiq_abort
from fastapi import APIRouter, HTTPException, status

from embedding_studio.api.api_v1.schemas.fine_tuning import (
    FineTuningTaskCreate,
    FineTuningTaskResponse,
)
from embedding_studio.context.app_context import context
from embedding_studio.models.fine_tuning import FineTuningStatus
from embedding_studio.workers.fine_tuning.worker import fine_tuning_worker

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/task",
    response_model=FineTuningTaskResponse,
    response_model_by_alias=False,
    response_model_exclude_none=True,
)
def create_fine_tuning_task(
    body: FineTuningTaskCreate,
) -> Any:
    """Create a new fine-tuning task.

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
    message = fine_tuning_worker.send(str(task.id))
    logger.debug(f"fine_tuning_worker message: {message}")
    task.broker_id = message.message_id
    context.fine_tuning_task.update(obj=task)
    return task


@router.get(
    "/task/{id}",
    response_model=FineTuningTaskResponse,
    response_model_by_alias=False,
    response_model_exclude_none=True,
)
def get_fine_tuning_task(
    id: str,
) -> Any:
    """Get details of a specific fine-tuning task.

    :param id: ID of the task.
    :return: Task details.
    """
    """
    Get details of a specific fine-tuning task.

    Args:
        id: str - ID of the task.

    Returns:
        FineTuningTaskResponse - Task details.
    """
    logger.debug(f"GET /task/{id}")
    task = context.fine_tuning_task.get(id)
    if task is not None:
        return task
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Task with ID `{id}` not found",
    )


@router.get(
    "/task",
    response_model=List[FineTuningTaskResponse],
    response_model_by_alias=False,
    response_model_exclude_none=True,
)
def get_fine_tuning_tasks(
    skip: int = 0,
    limit: int = 100,
    status: FineTuningStatus = None,
) -> Any:
    """Get a list of fine-tuning tasks.

    :param skip: Number of tasks to skip.
    :param limit: Number of tasks to return.
    :param status: Filter tasks by status.
    :return: List of tasks.
    """
    logger.debug(f"GET /task?skip={skip}%limit={limit}")
    if status is None:
        tasks = context.fine_tuning_task.get_all(skip=skip, limit=limit)
    else:
        tasks = context.fine_tuning_task.get_by_filter(
            {"status": status}, skip=skip, limit=limit
        )
    return tasks


@router.put(
    "/task/{id}/restart",
    response_model=FineTuningTaskResponse,
    response_model_by_alias=False,
    response_model_include={"id", "status"},
)
def restart_fine_tuning_task(
    id: str,
) -> Any:
    """Restart a fine-tuning task.

    :param id: ID of the task.
    :return: Restarted task details.
    """
    logger.debug(f"PUT /task/{id}/restart")
    task = context.fine_tuning_task.get(id)
    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task with ID `{id}` not found",
        )
    if task.status != FineTuningStatus.processing:
        task.status = "pending"
        task = context.fine_tuning_task.update(obj=task)
        message = fine_tuning_worker.send(str(task.id))
        logger.debug(f"fine_tuning_worker message: {message}")
        task = context.fine_tuning_task.update(
            obj=task, values={"broker_id": message.message_id}
        )
    return task


@router.put(
    "/task/{id}/cancel",
    response_model=FineTuningTaskResponse,
    response_model_by_alias=False,
    response_model_include={"id", "status"},
)
def cancel_fine_tuning_task(
    id: str,
) -> Any:
    """Cancel a fine-tuning task.

    :param id: ID of the task.
    :return: Canceled task details.
    """
    logger.debug(f"PUT /task/{id}/cancel")
    task = context.fine_tuning_task.get(id)
    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task with ID `{id}` not found",
        )
    dramatiq_abort(task.broker_id)
    task.status = "canceled"
    context.fine_tuning_task.update(obj=task)
    return task
