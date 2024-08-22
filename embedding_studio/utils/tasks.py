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
    return response_schema.model_validate(task.model_dump())


def create_task_helpers_router(
    task_crud: CRUDBase,
    response_model: Type[T],
    worker_func,
):
    router = APIRouter()

    def _get_task(task_id: str) -> Any:
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
        task = _get_task(task_id)
        dramatiq_abort(task.broker_id)
        task.status = TaskStatus.canceled
        task_crud.update(obj=task)
        return convert_to_response(task, response_model)

    return router
