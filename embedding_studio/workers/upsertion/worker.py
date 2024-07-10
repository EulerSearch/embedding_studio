import logging

import dramatiq

from embedding_studio.context.app_context import context
from embedding_studio.core.config import settings
from embedding_studio.db.redis import redis_broker
from embedding_studio.utils.dramatiq_middlewares import (
    ActionsOnStartMiddleware,
)
from embedding_studio.utils.initializer_actions import init_nltk
from embedding_studio.workers.upsertion.utils.upsert import handle_upsert

# Set up logging
logger = logging.getLogger(__name__)

redis_broker.add_middleware(ActionsOnStartMiddleware([init_nltk]))


@dramatiq.actor(
    queue_name="upsertion_worker",
    max_retries=settings.UPSERTION_WORKER_MAX_RETRIES,
    time_limit=settings.UPSERTION_WORKER_TIME_LIMIT,
)
def upsertion_worker(task_id: str):
    task = context.upsertion_task.get(id=task_id)
    handle_upsert(task)
