from fastapi import APIRouter

from embedding_studio.api.api_v1.internal_endpoints import (
    delete as internal_delete,
    inference_deployment_tasks,
    upsert as internal_upsert,
    vectordb,
)
from embedding_studio.api.api_v1.schemas.delete import DeletionTaskResponse
from embedding_studio.api.api_v1.schemas.upsert import UpsertionTaskResponse
from embedding_studio.context.app_context import context
from embedding_studio.utils import tasks
from embedding_studio.workers.upsertion.worker import (
    deletion_worker,
    upsertion_worker,
)


def add_internal_endpoints(api_router: APIRouter):
    # Use the create_task_helpers_router for internal deletion tasks
    internal_deletion_helpers_router = tasks.create_task_helpers_router(
        task_crud=context.deletion_task,
        response_model=DeletionTaskResponse,
        worker_func=deletion_worker,
    )
    api_router.include_router(
        internal_deletion_helpers_router,
        prefix="/internal/deletion-tasks",
        tags=["internal-deletion-tasks"],
    )
    api_router.include_router(
        internal_delete.router,  # This should now only contain the /run endpoint
        prefix="/internal/deletion-tasks",
        tags=["internal-deletion-tasks"],
    )

    # Use the create_task_helpers_router for internal upsertion tasks
    internal_upsertion_helpers_router = tasks.create_task_helpers_router(
        task_crud=context.upsertion_task,
        response_model=UpsertionTaskResponse,
        worker_func=upsertion_worker,
    )
    api_router.include_router(
        internal_upsertion_helpers_router,
        prefix="/internal/upsertion-tasks",
        tags=["internal-upsertion-tasks"],
    )
    api_router.include_router(
        internal_upsert.router,  # This should now only contain the /run endpoint
        prefix="/internal/upsertion-tasks",
        tags=["internal-upsertion-tasks"],
    )

    api_router.include_router(
        inference_deployment_tasks.router,
        prefix="/internal/inference-deployment",
        tags=["inference-deployment"],
    )
    api_router.include_router(
        vectordb.router,
        prefix="/internal/vectordb",
        tags=["vectordb"],
    )
