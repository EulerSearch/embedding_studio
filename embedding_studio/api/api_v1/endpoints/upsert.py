import logging
from typing import Any

from fastapi import APIRouter, HTTPException, status

from embedding_studio.api.api_v1.internal_schemas.upsert import (
    UpsertionTaskRunRequest as InternalUpsertionTaskRunRequest,
)
from embedding_studio.api.api_v1.schemas.upsert import (
    UpsertionTaskResponse,
    UpsertionTaskRunRequest,
)
from embedding_studio.context.app_context import context
from embedding_studio.utils.dramatiq_task_handler import create_and_send_task
from embedding_studio.utils.tasks import convert_to_response
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
    body: UpsertionTaskRunRequest,
) -> Any:
    """Create a new upsertion task.

    :param body: Request body.
    :return: Created task details.
    """
    logger.debug(f"POST /embeddings/upsertion-tasks/run: {body}")

    # Check if a task with the given task_id already exists
    if body.task_id:
        existing_task = context.upsertion_task.get(body.task_id)
        if existing_task:
            return convert_to_response(existing_task, UpsertionTaskResponse)

    collection = context.vectordb.get_blue_collection()
    collection_info = collection.get_info()
    task = context.upsertion_task.create(
        schema=InternalUpsertionTaskRunRequest(
            embedding_model_id=collection_info.embedding_model.id,
            fine_tuning_method=collection_info.embedding_model.name,
            items=body.items,
        ),
        return_obj=True,
        id=body.task_id,  # Use the provided task_id if available
    )

    # Use create_and_send_task instead of manual sending and updating
    updated_task = create_and_send_task(
        upsertion_worker, task, context.upsertion_task
    )

    if updated_task:
        return convert_to_response(updated_task, UpsertionTaskResponse)
    else:
        # If create_and_send_task returns None, it means there was an error
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create and send upsertion task",
        )
