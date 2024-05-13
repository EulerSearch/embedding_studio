from fastapi import APIRouter

from embedding_studio.api.api_v1.endpoints import (
    clickstream_client,
    clickstream_internal,
    fine_tuning,
    ping,
)
from embedding_studio.core.config import settings

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


if settings.OPEN_TEST_ENDPOINTS:
    from embedding_studio.api.api_v1.test_endpoints import (
        inference_deployment,
        vectordb,
    )

    api_router.include_router(
        inference_deployment.router,
        prefix="/inference-deployment",
        tags=["inference-deployment"],
    )
    api_router.include_router(
        vectordb.router,
        prefix="/vectordb",
        tags=["vectordb"],
    )

if settings.OPEN_MOCKED_ENDPOINTS:
    from embedding_studio.api.api_v1.mocked_endpoints import mocked_fine_tuning

    api_router.include_router(
        mocked_fine_tuning.router,
        prefix="/mocked-fine-tuning",
        tags=["mocked-fine-tuning"],
    )
