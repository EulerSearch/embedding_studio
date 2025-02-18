from fastapi import APIRouter

from embedding_studio.api.api_v1.endpoints import (
    clickstream_client,
    clickstream_internal,
    delete,
    fine_tuning,
    ping,
    query_parsing,
    similarity_search,
    suggesting,
    upsert,
)
from embedding_studio.api.api_v1.internal_api import add_internal_endpoints
from embedding_studio.api.api_v1.mocked_api import add_mocked_endpoints
from embedding_studio.api.api_v1.schemas.delete import DeletionTaskResponse
from embedding_studio.api.api_v1.schemas.upsert import UpsertionTaskResponse
from embedding_studio.context.app_context import context
from embedding_studio.core.config import settings
from embedding_studio.utils import tasks
from embedding_studio.workers.upsertion.worker import (
    deletion_worker,
    upsertion_worker,
)

api_router = APIRouter()
api_router.include_router(ping.router, tags=["ping"])

api_router.include_router(
    query_parsing.router, prefix="/parse-query", tags=["query_parsing"]
)
api_router.include_router(
    similarity_search.router, prefix="/embeddings", tags=["similarity_search"]
)
api_router.include_router(
    suggesting.router, prefix="/suggesting", tags=["suggesting"]
)

# Use the create_task_helpers_router for deletion tasks
deletion_helpers_router = tasks.create_task_helpers_router(
    task_crud=context.deletion_task,
    response_model=DeletionTaskResponse,
    worker_func=deletion_worker,
)
api_router.include_router(
    deletion_helpers_router,
    prefix="/embeddings/deletion-tasks",
    tags=["deletion-tasks"],
)
api_router.include_router(
    delete.router, prefix="/embeddings/deletion-tasks", tags=["deletion-tasks"]
)

# Use the create_task_helpers_router for upsertion tasks
upsertion_helpers_router = tasks.create_task_helpers_router(
    task_crud=context.upsertion_task,
    response_model=UpsertionTaskResponse,
    worker_func=upsertion_worker,
)
api_router.include_router(
    upsertion_helpers_router,
    prefix="/embeddings/upsertion-tasks",
    tags=["upsertion-tasks"],
)
api_router.include_router(
    upsert.router,  # This should now only contain the /run endpoint
    prefix="/embeddings/upsertion-tasks",
    tags=["upsertion-tasks"],
)

api_router.include_router(
    fine_tuning.router, prefix="/fine-tuning", tags=["fine-tuning"]
)
api_router.include_router(
    clickstream_client.router, prefix="/clickstream", tags=["clickstream"]
)
api_router.include_router(
    clickstream_internal.router,
    prefix="/clickstream/internal",
    tags=["clickstream"],
)

if settings.OPEN_INTERNAL_ENDPOINTS:
    add_internal_endpoints(api_router)

if settings.OPEN_MOCKED_ENDPOINTS:
    add_mocked_endpoints(api_router)
