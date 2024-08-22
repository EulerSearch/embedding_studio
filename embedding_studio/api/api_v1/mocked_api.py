from fastapi import APIRouter

from embedding_studio.api.api_v1.mocked_endpoints import mocked_fine_tuning


def add_mocked_endpoints(api_router: APIRouter):
    api_router.include_router(
        mocked_fine_tuning.router,
        prefix="/mocked/fine-tuning",
        tags=["mocked-fine-tuning"],
    )
