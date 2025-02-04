import logging
from typing import Any

from fastapi import APIRouter, HTTPException, status

from embedding_studio.api.api_v1.schemas.delete import (
    DeletionTaskResponse,
    DeletionTaskRunRequest,
)
from embedding_studio.context.app_context import context
from embedding_studio.models.delete import DeletionTaskCreateSchema
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
    logger.debug(f"POST /embeddings/deletion-tasks/run: {body}")

    # Check if a task with the given task_id already exists
    if body.task_id:
        existing_task = context.deletion_task.get(body.task_id)
        if existing_task:
            return convert_to_response(existing_task, DeletionTaskResponse)

    collection = context.vectordb.get_blue_collection()
    collection_info = collection.get_info()
    task = context.deletion_task.create(
        schema=DeletionTaskCreateSchema(
            embedding_model_id=collection_info.embedding_model.id,
            fine_tuning_method=collection_info.embedding_model.name,
            object_ids=body.object_ids,
        ),
        return_obj=True,
        id=body.task_id,  # Use the provided task_id if available
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


@router.post(
    "/categories/run",
    response_model=DeletionTaskResponse,
    response_model_by_alias=False,
    response_model_exclude_none=True,
)
def create_categories_deletion_task(
    body: DeletionTaskRunRequest,
) -> Any:
    """Create a new categories deletion task.

    :param body: Request body.
    :return: Created task details.
    """
    logger.debug(f"POST /embeddings/deletion-tasks/categories/run: {body}")

    # Check if a task with the given task_id already exists
    if body.task_id:
        existing_task = context.deletion_task.get(body.task_id)
        if existing_task:
            return convert_to_response(existing_task, DeletionTaskResponse)

    collection = context.categories_vectordb.get_blue_collection()

    collection_info = collection.get_info()
    task = context.deletion_task.create(
        schema=DeletionTaskCreateSchema(
            embedding_model_id=collection_info.embedding_model.id,
            fine_tuning_method=collection_info.embedding_model.name,
            object_ids=body.object_ids,
        ),
        return_obj=True,
        id=body.task_id,  # Use the provided task_id if available
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
            detail="Failed to create and send categories deletion task",
        )
