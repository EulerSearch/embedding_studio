import logging
from typing import Any, List

from fastapi import APIRouter, HTTPException, status

from embedding_studio.crud.fine_tuning import fine_tuning_task
from embedding_studio.schemas.fine_tuning import (
    FineTuningTaskCreate,
    FineTuningTaskInDb,
)
from embedding_studio.workers.fine_tuning import fine_tuning_worker

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/create",
    status_code=status.HTTP_201_CREATED,
    response_model=FineTuningTaskInDb,
    response_model_by_alias=False,
)
def create_fine_tuning_task(
    body: FineTuningTaskCreate,
) -> Any:
    # TODO: add idempotency key
    logger.debug(f"/create: {body}")
    task = fine_tuning_task.create(schema=body, return_obj=True)
    fine_tuning_worker.send(str(task.id))
    return task


@router.get(
    "/get/{id}",
    response_model=FineTuningTaskInDb,
    response_model_by_alias=False,
)
def get_fine_tuning_task(
    id: str,
) -> Any:
    logger.debug(f"/get/{id}")
    task = fine_tuning_task.get(id)
    if task is not None:
        return task
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Task with ID `{id}` not found",
    )


@router.get(
    "/get",
    response_model=List[FineTuningTaskInDb],
    response_model_by_alias=False,
)
def get_fine_tuning_tasks(
    skip: int = 0,
    limit: int = 100,
) -> Any:
    logger.debug(f"/get: skip={skip}, limit={limit}")
    tasks = fine_tuning_task.get_all(skip=skip, limit=limit)
    return tasks
