import logging
from typing import Any

from fastapi import APIRouter, HTTPException, status

from embedding_studio.api.api_v1.schemas.inference_deployment import (
    DeploymentInfo,
    DeploymentRequest,
    DeploymentResponse,
    GreenDeploymentRequest,
    GreenDeploymentAfterFineTuningRequest,
)
from embedding_studio.context.app_context import context
from embedding_studio.workers.inference.worker import deployment_worker

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post(
    "/deploy/green",
    response_model=DeploymentResponse,
    response_model_by_alias=False,
    response_model_exclude_none=True,
)
def green_deployment(
    body: GreenDeploymentRequest,
) -> Any:
    """Deploy a model with the green deployment strategy.

    :param body: The deployment request body.
    :return: The result of the deployment process.
    """
    logger.debug(f"POST /deploy/green: {body}")


    info = DeploymentInfo(
        fine_tuning_method=body.fine_tuning_method,
        stage="green",
    )
    deployment_task = context.deployment_task.create(
        schema=info, return_obj=True
    )
    message = deployment_worker.send(str(deployment_task.id))
    logger.debug(f"Green deployment message: {message}")
    deployment_task.broker_id = message.message_id
    context.deployment_task.update(obj=deployment_task)
    return deployment_task


@router.post(
    "/deploy/green/from-fine-tuning",
    response_model=DeploymentResponse,
    response_model_by_alias=False,
    response_model_exclude_none=True,
)
def from_fine_tunnig_green_deployment(
    body: GreenDeploymentAfterFineTuningRequest,
) -> Any:
    """Deploy a model with the green deployment strategy after fine-tuning.

    :param body: The deployment request body.
    :return: The result of the deployment process.
    """
    logger.debug(f"POST /deploy/green: {body}")
    fine_tuning_task = context.fine_tuning_task.get(body.fine_tuning_task_id)
    if fine_tuning_task is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Fine-tuning task with ID `{body.fine_tuning_task_id}` not found",
        )

    info = DeploymentInfo(
        fine_tuning_method=fine_tuning_task.fine_tuning_method,
        stage="green",
    )
    deployment_task = context.deployment_task.create(
        schema=info, return_obj=True
    )
    message = deployment_worker.send(str(deployment_task.id))
    logger.debug(f"Green deployment message: {message}")
    deployment_task.broker_id = message.message_id
    context.deployment_task.update(obj=deployment_task)
    return deployment_task


@router.get(
    "/deploy/green/{id}",
    response_model=DeploymentResponse,
    response_model_by_alias=False,
    response_model_exclude_none=True,
)
def get_green_deployment_task(
    id: str,
) -> Any:
    """Get details of a specific green deployment.

    :param id: ID of the task.
    :return: Task details.
    """
    task = context.deployment_task.get(id)
    if task is not None and task.stage == "green":
        return task

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Task with ID `{id}` on green stage is not found",
    )


@router.post(
    "/deploy/blue",
    response_model=DeploymentResponse,
    response_model_by_alias=False,
    response_model_exclude_none=True,
)
def blue_deployment(
    body: DeploymentRequest,
) -> Any:
    """Deploy a model with the blue deployment strategy.

    :param body: The deployment request body.
    :return: The result of the deployment process.
    """
    logger.debug(f"POST /deploy/blue: {body}")
    deployment_task = context.deployment_task.get(
        body.task_id
    )
    if deployment_task is not None:
        if deployment_task.stage == "blue":
            return deployment_task

        elif deployment_task.stage == "green" and deployment_task.status != "done":
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Task with ID `{body.task_id}` is on green stage, but not done.",
            )

        elif deployment_task.stage == "revert":
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Task with ID `{body.task_id}` is on revert stage",
            )

    else:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task with ID `{body.task_id}` not found",
        )

    deployment_task.stage = "blue"
    context.deployment_task.update(obj=deployment_task)
    message = deployment_worker.send(str(deployment_task.id))
    logger.debug(f"Blue deployment message: {message}")
    deployment_task.broker_id = message.message_id

    return deployment_task


@router.get(
    "/deploy/blue/{id}",
    response_model=DeploymentResponse,
    response_model_by_alias=False,
    response_model_exclude_none=True,
)
def get_blue_deployment_task(
    id: str,
) -> Any:
    """Get details of a specific blue deployment.

    :param id: ID of the task.
    :return: Task details.
    """
    task = context.deployment_task.get(id)
    if task is not None and task.stage == "blue":
        return task

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Task with ID `{id}` on blue stage is not found",
    )


@router.post(
    "/deploy/revert",
    response_model=DeploymentResponse,
    response_model_by_alias=False,
    response_model_exclude_none=True,
)
def revert_deployment(
    body: DeploymentRequest,
) -> Any:
    """Revert to a previous deployment.

    :param body: The deployment request body.
    :return: The result of the deployment process.
    """
    logger.debug(f"POST /deploy/revert: {body}")
    deployment_task = context.deployment_task.get(
        body.task_id
    )
    if deployment_task is not None:
        if deployment_task.stage == "revert":
            return deployment_task

        elif deployment_task.stage == "green":
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Task with ID `{body.task_id}` is on green stage.",
            )

        elif deployment_task.stage == "blue" and deployment_task.status != "done":
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Task with ID `{body.task_id}` is on blue stage, but not done.",
            )

    else:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task with ID `{body.task_id}` not found",
        )

    deployment_task.stage = "revert"
    context.deployment_task.update(obj=deployment_task)
    message = deployment_worker.send(str(deployment_task.id))
    logger.debug(f"Revert deployment message: {message}")
    deployment_task.broker_id = message.message_id
    return deployment_task


@router.get(
    "/deploy/revert/{id}",
    response_model=DeploymentResponse,
    response_model_by_alias=False,
    response_model_exclude_none=True,
)
def get_blue_deployment_task(
    id: str,
) -> Any:
    """Get details of a specific deployment revert.

    :param id: ID of the task.
    :return: Task details.
    """
    task = context.deployment_task.get(id)
    if task is not None and task.stage == "revert":
        return task

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Task with ID `{id}` on revert stage is not found",
    )
