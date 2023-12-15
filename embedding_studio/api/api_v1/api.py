from fastapi import APIRouter

from embedding_studio.api.api_v1.endpoints import (
    clickstream_client,
    clickstream_internal,
    fine_tuning,
    ping,
)

api_router = APIRouter()
api_router.include_router(ping.router, tags=["ping"])
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
