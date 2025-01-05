import logging
from typing import Any

from fastapi import APIRouter, HTTPException, status

from embedding_studio.api.api_v1.internal_schemas.inference_deployment_tasks import (
    ModelDeletionRequest,
    ModelDeletionResponse,
    ModelDeploymentRequest,
    ModelDeploymentResponse,
)
from embedding_studio.context.app_context import context
from embedding_studio.utils.dramatiq_task_handler import create_and_send_task
from embedding_studio.utils.tasks import convert_to_response
from embedding_studio.workers.inference.worker import (
    model_deletion_worker,
    model_deployment_worker,
)

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post(
    "/deploy",
    response_model=ModelDeploymentResponse,
    response_model_by_alias=False,
    response_model_exclude_none=True,
)
def deploy(
    body: ModelDeploymentRequest,
) -> Any:
    """Deploy a model with the green deployment strategy.

    :param body: The deployment request body.
    :return: The result of the deployment process.
    """
    logger.debug(f"POST /deploy: {body}")
    deployment_task = context.model_deployment_task.create(
        schema=body, return_obj=True
    )

    # Use create_and_send_task instead of manual sending and updating
    updated_task = create_and_send_task(
        model_deployment_worker, deployment_task, context.model_deployment_task
    )

    if updated_task:
        return convert_to_response(updated_task, ModelDeploymentResponse)
    else:
        # If create_and_send_task returns None, it means there was an error
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create and send model deployment task",
        )


@router.get(
    "/deploy/{task_id}",
    response_model=ModelDeploymentResponse,
    response_model_by_alias=False,
    response_model_exclude_none=True,
)
def get_deploy_task(
    task_id: str,
) -> Any:
    """Get details of a specific deployment.

    :param task_id: ID of the task.
    :return: Task details.
    """
    task = context.model_deployment_task.get(task_id)
    if task is not None:
        return convert_to_response(task, ModelDeploymentResponse)

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Deployment task with ID `{task_id}` is not found",
    )


@router.get(
    "/deploy-status/{embedding_model_id}",
    response_model=ModelDeploymentResponse,
    response_model_by_alias=False,
    response_model_exclude_none=True,
)
def get_model_deploy_status(
    embedding_model_id: str,
) -> Any:
    """Get details of a specific deployment.

    :param embedding_model_id: ID of the model.
    :return: Task details.
    """
    task = context.model_deployment_task.get_by_model_id(embedding_model_id)
    if task is not None:
        return convert_to_response(task, ModelDeploymentResponse)

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Deployment task with embedding model ID `{embedding_model_id}` is not found",
    )


@router.post(
    "/delete",
    response_model=ModelDeletionResponse,
    response_model_by_alias=False,
    response_model_exclude_none=True,
)
def delete(
    body: ModelDeletionRequest,
) -> Any:
    """Delete a model.

    :param body: The deletion request body.
    :return: The result of the deletion process.
    """
    logger.debug(f"POST /delete: {body}")
    deletion_task = context.model_deletion_task.create(
        schema=body, return_obj=True
    )

    # Use create_and_send_task instead of manual sending and updating
    updated_task = create_and_send_task(
        model_deletion_worker, deletion_task, context.model_deletion_task
    )

    if updated_task:
        return convert_to_response(updated_task, ModelDeletionResponse)
    else:
        # If create_and_send_task returns None, it means there was an error
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create and send model deletion task",
        )


@router.get(
    "/delete/{task_id}",
    response_model=ModelDeletionResponse,
    response_model_by_alias=False,
    response_model_exclude_none=True,
)
def get_delete_task(
    task_id: str,
) -> Any:
    """Get details of a specific deployment revert.

    :param task_id: ID of the task.
    :return: Task details.
    """
    task = context.model_deployment_task.get(task_id)
    if task is not None:
        return convert_to_response(task, ModelDeletionResponse)

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Deletion task with ID `{task_id}` is not found",
    )


@router.get(
    "/delete-status/{embedding_model_id}",
    response_model=ModelDeletionResponse,
    response_model_by_alias=False,
    response_model_exclude_none=True,
)
def get_model_delete_status(
    embedding_model_id: str,
) -> Any:
    """Get details of a specific deployment revert.

    :param embedding_model_id: ID of the task.
    :return: Task details.
    """
    task = context.model_deployment_task.get_by_model_id(embedding_model_id)
    if task is not None:
        return convert_to_response(task, ModelDeletionResponse)

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Deletion task with embedding model ID `{embedding_model_id}` is not found",
    )
