import logging

import dramatiq

from embedding_studio.core.config import settings
from embedding_studio.db.redis import redis_broker
from embedding_studio.utils.dramatiq_middlewares import (
    ActionsOnStartMiddleware,
)
from embedding_studio.utils.initializer_actions import init_nltk
from embedding_studio.workers.inference.utils.deletion import handle_deletion
from embedding_studio.workers.inference.utils.deployment import (
    handle_deployment,
)
from embedding_studio.workers.inference.utils.init_model_repo import (
    OnStartMiddleware,
)

# Set up logging
logger = logging.getLogger(__name__)

redis_broker.add_middleware(ActionsOnStartMiddleware([init_nltk]))
redis_broker.add_middleware(OnStartMiddleware())


@dramatiq.actor(
    queue_name="deployment_worker",
    max_retries=settings.INFERENCE_WORKER_MAX_RETRIES,
    time_limit=settings.INFERENCE_WORKER_TIME_LIMIT,
)
def deployment_worker(task_id: str):
    handle_deployment(task_id)


@dramatiq.actor(
    queue_name="deletion_worker",
    max_retries=settings.INFERENCE_WORKER_MAX_RETRIES,
    time_limit=settings.INFERENCE_WORKER_TIME_LIMIT,
)
def deletion_worker(task_id: str):
    handle_deletion(task_id)
