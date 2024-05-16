import logging
from typing import Any

from fastapi import APIRouter, HTTPException, status

from embedding_studio.api.api_v1.schemas.inference_deployment_tasks import (
    ModelDeletionRequest,
    ModelDeletionResponse,
    ModelDeploymentRequest,
    ModelDeploymentResponse,
)
from embedding_studio.context.app_context import context
from embedding_studio.workers.inference.worker import (
    deletion_worker,
    deployment_worker,
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
    deployment_task = context.deployment_task.create(
        schema=body, return_obj=True
    )
    message = deployment_worker.send(str(deployment_task.id))
    logger.debug(f"Green deployment message: {message}")
    deployment_task.broker_id = message.message_id
    context.deployment_task.update(obj=deployment_task)
    return deployment_task


@router.get(
    "/deploy/{embedding_model_id}",
    response_model=ModelDeploymentResponse,
    response_model_by_alias=False,
    response_model_exclude_none=True,
)
def get_deploy_task(
    embedding_model_id: str,
) -> Any:
    """Get details of a specific green deployment.

    :param id: ID of the model.
    :return: Task details.
    """
    task = context.deployment_task.get_by_model_id(embedding_model_id)
    if task is not None:
        return task

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

    :param body: The deployment request body.
    :return: The result of the deployment process.
    """
    logger.debug(f"POST /delete: {body}")
    deletion_task = context.deletion_task.create(schema=body, return_obj=True)
    message = deletion_worker.send(str(deletion_task.id))
    logger.debug(f"Revert deployment message: {message}")
    deletion_task.broker_id = message.message_id
    context.deletion_task.update(obj=deletion_task)
    return deletion_task


@router.get(
    "/delete/{embedding_model_id}",
    response_model=ModelDeletionResponse,
    response_model_by_alias=False,
    response_model_exclude_none=True,
)
def get_delete_task(
    embedding_model_id: str,
) -> Any:
    """Get details of a specific deployment revert.

    :param id: ID of the task.
    :return: Task details.
    """
    task = context.deployment_task.get_by_model_id(embedding_model_id)
    if task is not None:
        return task

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Deletion task with embedding model ID `{embedding_model_id}` is not found",
    )
