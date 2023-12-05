from fastapi import APIRouter

from embedding_studio.api.api_v1.endpoints import fine_tuning, ping

api_router = APIRouter()
api_router.include_router(ping.router, tags=["ping"])
api_router.include_router(
    fine_tuning.router, prefix="/fine-tuning", tags=["fine-tuning"]
)
