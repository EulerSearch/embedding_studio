import logging
from typing import Any

from fastapi import APIRouter, HTTPException, status

from embedding_studio.api.api_v1.internal_schemas.delete import (
    DeletionTaskResponse,
    DeletionTaskRunRequest,
)
from embedding_studio.context.app_context import context
from embedding_studio.utils.dramatiq_task_handler import create_and_send_task
from embedding_studio.utils.tasks import convert_to_response
from embedding_studio.workers.upsertion.worker import deletion_worker

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/run",
    response_model=DeletionTaskResponse,
    response_model_by_alias=False,
    response_model_exclude_none=True,
)
def create_deletion_task(
    body: DeletionTaskRunRequest,
) -> Any:
    """Create a new deletion task.

    :param body: Request body.
    :return: Created task details.
    """
    logger.debug(f"POST /internal/deletion-task/run: {body}")

    # Check if a task with the given task_id already exists
    if body.task_id:
        existing_task = context.deletion_task.get(body.task_id)
        if existing_task:
            return convert_to_response(existing_task, DeletionTaskResponse)

    # Create new task
    task = context.deletion_task.create(
        schema=body, return_obj=True, id=body.task_id
    )

    # Use create_and_send_task instead of manual sending and updating
    updated_task = create_and_send_task(
        deletion_worker, task, context.deletion_task
    )

    if updated_task:
        return convert_to_response(updated_task, DeletionTaskResponse)
    else:
        # If create_and_send_task returns None, it means there was an error
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create and send deletion task",
        )
