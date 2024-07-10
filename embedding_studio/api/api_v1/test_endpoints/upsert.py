import logging
from typing import Any, List

from dramatiq_abort import abort as dramatiq_abort
from fastapi import APIRouter, HTTPException, status

from embedding_studio.api.api_v1.schemas.upsert import (
    UpsertionStatus,
    UpsertionTaskCancelResponse,
    UpsertionTaskCreateRequest,
    UpsertionTaskDeleteResponse,
    UpsertionTaskResponse,
)
from embedding_studio.context.app_context import context
from embedding_studio.workers.upsertion.worker import upsertion_worker

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/run",
    response_model=UpsertionTaskResponse,
    response_model_by_alias=False,
    response_model_exclude_none=True,
)
def create_upsertion_task(
    body: UpsertionTaskCreateRequest,
) -> Any:
    """Create a new upserion task.

    :param body: Request body.
    :return: Created task details.
    """
    # TODO: check if task with the same batch_id and params already exists
    logger.debug(f"POST /test-upsertion-task/run: {body}")
    task = context.upsertion_task.create(schema=body, return_obj=True)
    message = upsertion_worker.send(str(task.id))
    logger.debug(f"upsertion_worker message: {message}")
    task.broker_id = message.message_id
    context.upsertion_task.update(obj=task)
    return task


@router.get(
    "/info",
    response_model=UpsertionTaskResponse,
    response_model_by_alias=False,
    response_model_exclude_none=True,
)
def get_upsertion_task(
    id: str,
) -> Any:
    """Get details of a specific upsertion task.

    :param id: ID of the task.
    :return: Task details.
    """
    logger.debug(f"GET /test-upsertion-task/info/{id}")
    task = context.upsertion_task.get(id)
    if task is not None:
        return task
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Task with ID `{id}` not found",
    )


@router.get(
    "/list",
    response_model=List[UpsertionTaskResponse],
    response_model_by_alias=False,
    response_model_exclude_none=True,
)
def get_upsertion_tasks(
    skip: int = 0,
    limit: int = 100,
    status: UpsertionStatus = None,
) -> Any:
    """Get a list of upsertion tasks.

    :param skip: Number of tasks to skip.
    :param limit: Number of tasks to return.
    :param status: Filter tasks by status.
    :return: List of tasks.
    """
    logger.debug(f"GET /test-upsertion-task/list?skip={skip}%limit={limit}")
    if status is None:
        tasks = context.upsertion_task.get_all(skip=skip, limit=limit)
    else:
        tasks = context.upsertion_task.get_by_filter(
            {"status": status}, skip=skip, limit=limit
        )
    return tasks


@router.put(
    "/restart",
    response_model=UpsertionTaskDeleteResponse,
    response_model_by_alias=False,
    response_model_exclude_none=True,
)
def restart_upsertion_task(
    id: str,
) -> Any:
    """Restart an upsertion task.

    :param id: ID of the task.
    :return: Restarted task details.
    """
    logger.debug(f"PUT /test-upsertion-task/restart/{id}")
    task = context.upsertion_task.get(id)
    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task with ID `{id}` not found",
        )
    if task.status != UpsertionStatus.processing:
        task.status = UpsertionStatus.pending
        task = context.upsertion_task.update(obj=task)
        message = upsertion_worker.send(str(task.id))
        logger.debug(f"upsertion_worker message: {message}")
        task = context.upsertion_worker.update(
            obj=task, values={"broker_id": message.message_id}
        )
    return task


@router.put(
    "/cancel",
    response_model=UpsertionTaskCancelResponse,
    response_model_by_alias=False,
    response_model_exclude_none=True,
)
def cancel_upsertion_task(
    id: str,
) -> Any:
    """Cancel an upsertion task.

    :param id: ID of the task.
    :return: Canceled task details.
    """
    logger.debug(f"PUT /test-upsertion-task/cancel/{id}")
    task = context.upsertion_task.get(id)
    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task with ID `{id}` not found",
        )
    dramatiq_abort(task.broker_id)
    task.status = UpsertionStatus.canceled
    context.upsertion_task.update(obj=task)
    return task
