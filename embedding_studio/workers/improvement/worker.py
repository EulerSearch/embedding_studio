import logging

import dramatiq

from embedding_studio.context.app_context import context
from embedding_studio.core.config import settings
from embedding_studio.db.redis import redis_broker
from embedding_studio.models.task import TaskStatus
from embedding_studio.utils.dramatiq_middlewares import (
    ActionsOnStartMiddleware,
)
from embedding_studio.utils.initializer_actions import (
    init_background_scheduler,
    init_nltk,
)
from embedding_studio.workers.improvement.utils.handle_improvement import (
    handle_improvement,
)

# Set up logging
logger = logging.getLogger(__name__)

redis_broker.add_middleware(ActionsOnStartMiddleware([init_nltk]))
redis_broker.add_middleware(
    ActionsOnStartMiddleware(
        [
            lambda: init_background_scheduler(
                improvement_worker.send, settings.IMPROVEMENT_SECONDS_INTERVAL
            )
        ]
    )
)


@dramatiq.actor(
    queue_name="improvement_worker",
    max_retries=settings.IMPROVEMENT_WORKER_MAX_RETRIES,
    time_limit=settings.IMPROVEMENT_WORKER_TIME_LIMIT,
)
def improvement_worker():
    sessions_for_improvement = []
    skip = 0

    while True:
        # Fetch a batch of matching tasks
        batch = context.sessions_for_improvement.get_by_filter(
            {"status": TaskStatus.pending}, skip=skip, limit=100
        )
        sessions_for_improvement += batch

        # If fewer results than the limit, it’s the last page
        if len(batch) < 100:
            break

        skip += 100

    if len(sessions_for_improvement) == 0:
        logger.info("No improvement sessions found")
        return

    logger.info(
        f"{len(sessions_for_improvement)} sessions for improvement found"
    )
    for session in sessions_for_improvement:
        session.status = TaskStatus.processing

        context.sessions_for_improvement.update(obj=session)

    handle_improvement(sessions_for_improvement)
