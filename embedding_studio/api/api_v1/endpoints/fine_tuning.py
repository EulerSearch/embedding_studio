import logging
from typing import Any, List

from dramatiq_abort import abort as dramatiq_abort
from fastapi import APIRouter, HTTPException, status

from embedding_studio.api.api_v1.schemas.fine_tuning import (
    FineTuningTaskResponse,
    FineTuningTaskRunRequest,
)
from embedding_studio.context.app_context import context
from embedding_studio.models.task import TaskStatus
from embedding_studio.utils.dramatiq_task_handler import create_and_send_task
from embedding_studio.utils.plugin_utils import is_basic_plugin
from embedding_studio.utils.tasks import convert_to_response
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
    body: FineTuningTaskRunRequest,
) -> Any:
    """Create a new fine-tuning task.

    :param body: Request body.
    :return: Created task details.
    """
    logger.debug(f"POST /task: {body}")

    plugin = context.plugin_manager.get_plugin(body.fine_tuning_method)
    if is_basic_plugin(plugin):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Fine-tuning method for not basic models is not supported",
        )

    # Check for existing task using idempotency key
    if body.idempotency_key:
        existing_task = context.fine_tuning_task.get_by_idempotency_key(
            body.idempotency_key
        )
        if existing_task is not None:
            return convert_to_response(existing_task, FineTuningTaskResponse)

    # TODO: check if task with the same batch_id and params already exists

    # Create new task
    task = context.fine_tuning_task.create(schema=body, return_obj=True)

    # Use create_and_send_task instead of manual sending and updating
    updated_task = create_and_send_task(
        fine_tuning_worker, task, context.fine_tuning_task
    )

    if updated_task:
        return convert_to_response(updated_task, FineTuningTaskResponse)
    else:
        # If create_and_send_task returns None, it means there was an error
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create and send fine-tuning task",
        )


# TODO: use create_task_helpers_router


@router.get(
    "/task",
    response_model=List[FineTuningTaskResponse],
    response_model_by_alias=False,
    response_model_exclude_none=True,
)
def get_fine_tuning_tasks(
    offset: int = 0,
    limit: int = 100,
    status: TaskStatus = None,
) -> Any:
    """Get a list of fine-tuning tasks.

    :param offset: Number of tasks to skip.
    :param limit: Number of tasks to return.
    :param status: Filter tasks by status.
    :return: List of tasks.
    """
    logger.debug(f"GET /task?offset={offset}%limit={limit}")
    if status is None:
        tasks = context.fine_tuning_task.get_all(skip=offset, limit=limit)
    else:
        tasks = context.fine_tuning_task.get_by_filter(
            {"status": status}, skip=offset, limit=limit
        )
    return [
        convert_to_response(task, FineTuningTaskResponse) for task in tasks
    ]


@router.put(
    "/task/{id}/restart",
    response_model=FineTuningTaskResponse,
    response_model_by_alias=False,
    response_model_exclude_none=True,
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
    if task.status != TaskStatus.processing:
        task.status = "pending"
        task = context.fine_tuning_task.update(obj=task)
        message = fine_tuning_worker.send(str(task.id))
        logger.debug(f"fine_tuning_worker message: {message}")
        task = context.fine_tuning_task.update(
            obj=task, values={"broker_id": message.message_id}
        )
    return convert_to_response(task, FineTuningTaskResponse)


@router.put(
    "/task/{id}/cancel",
    response_model=FineTuningTaskResponse,
    response_model_by_alias=False,
    response_model_exclude_none=True,
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
    return convert_to_response(task, FineTuningTaskResponse)
