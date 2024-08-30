import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from embedding_studio.api.api_v1.api import api_router
from embedding_studio.core.config import settings
from embedding_studio.utils.initializer_actions import (
    init_nltk,
    init_plugin_manager,
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # pre-launch actions
    init_nltk()
    init_plugin_manager()
    yield
    # post actions


origins = ["*"]
app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(api_router, prefix=settings.API_V1_STR)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=5000, log_config="log_config.yaml")
