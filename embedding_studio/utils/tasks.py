from typing import Any, List, Type, TypeVar

from dramatiq_abort import abort as dramatiq_abort
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from embedding_studio.api.api_v1.schemas.task import (
    BaseTaskResponse,
    TaskStatus,
)
from embedding_studio.data_access.mongo.crud_base import CRUDBase
from embedding_studio.models.task import BaseTaskInDb
from embedding_studio.utils.dramatiq_task_handler import create_and_send_task

T = TypeVar("T", bound=BaseModel)


def convert_to_response(
    task: BaseTaskInDb, response_schema: Type[BaseTaskResponse]
) -> BaseTaskResponse:
    """
    Convert a database task object to a response model.

    This function transforms a task database model into a response schema model
    for API responses.

    :param task: The database task object to convert
    :param response_schema: The response model class to convert to
    :return: A response model instance populated with task data
    """
    return response_schema.model_validate(task.model_dump())


def create_task_helpers_router(
    task_crud: CRUDBase,
    response_model: Type[T],
    worker_func,
):
    """
    Create a FastAPI router with task management endpoints.

    This factory function generates a router with the following endpoints:
    - GET /info: Get information about a specific task
    - GET /list: List tasks with optional filtering
    - PUT /restart: Restart a task
    - PUT /cancel: Cancel a task

    :param task_crud: CRUD manager for task operations
    :param response_model: The response model to use for API responses
    :param worker_func: The worker function to use for task execution
    :return: A FastAPI router with task management endpoints
    """

    router = APIRouter()

    def _get_task(task_id: str) -> Any:
        """
        Internal helper to get a task by ID and convert it to a response model.

        Raises an HTTP 404 exception if the task is not found.

        :param task_id: The ID of the task to retrieve
        :return: The task data as a response model
        """
        task = task_crud.get(task_id)
        if task is not None:
            return convert_to_response(task, response_model)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task with ID `{task_id}` not found",
        )

    @router.get(
        "/info",
        response_model=response_model,
        response_model_by_alias=False,
        response_model_exclude_none=True,
    )
    def get_task(task_id: str) -> Any:
        """
        Get information about a specific task.

        :param task_id: The ID of the task to retrieve
        :return: The task data as a response model
        """
        task = _get_task(task_id)
        return response_model.model_validate(task)

    @router.get(
        "/list",
        response_model=List[response_model],
        response_model_by_alias=False,
        response_model_exclude_none=True,
    )
    def get_tasks(
        offset: int = 0,
        limit: int = 100,
        status: TaskStatus = None,
    ) -> Any:
        """
        List tasks with optional filtering by status.

        :param offset: Number of items to skip (for pagination)
        :param limit: Maximum number of items to return
        :param status: Optional task status to filter by
        :return: A list of task data as response models
        """
        if status is None:
            tasks = task_crud.get_all(skip=offset, limit=limit)
        else:
            tasks = task_crud.get_by_filter(
                {"status": status}, skip=offset, limit=limit
            )
        return [convert_to_response(task, response_model) for task in tasks]

    @router.put(
        "/restart",
        response_model=response_model,
        response_model_by_alias=False,
        response_model_exclude_none=True,
    )
    def restart_task(task_id: str) -> Any:
        """
        Restart a task by ID.

        This endpoint sets the task status to pending and re-sends it to the worker
        if it is not currently processing.

        :param task_id: The ID of the task to restart
        :return: The updated task data as a response model
        """
        task = task_crud.get(task_id)
        if not task:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Task with ID `{task_id}` not found",
            )
        if task.status != TaskStatus.processing:
            task.status = TaskStatus.pending
            task = task_crud.update(obj=task)

            # Use create_and_send_task instead of manual sending and updating
            updated_task = create_and_send_task(worker_func, task, task_crud)

            if updated_task:
                return convert_to_response(updated_task, response_model)
            else:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to restart task with ID `{task_id}`",
                )
        return convert_to_response(task, response_model)

    @router.put(
        "/cancel",
        response_model=response_model,
        response_model_by_alias=False,
        response_model_exclude_none=True,
    )
    def cancel_task(task_id: str) -> Any:
        """
        Cancel a task by ID.

        This endpoint aborts the task's execution in the message broker and
        updates its status to canceled.

        :param task_id: The ID of the task to cancel
        :return: The updated task data as a response model
        """
        task = _get_task(task_id)
        dramatiq_abort(task.broker_id)
        task.status = TaskStatus.canceled
        task_crud.update(obj=task)
        return convert_to_response(task, response_model)

    return router
