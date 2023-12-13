import json
import logging

import dramatiq

from embedding_studio.core.config import settings
from embedding_studio.core.plugin import PluginManager
from embedding_studio.crud.fine_tuning import fine_tuning_task
from embedding_studio.db.redis import redis_broker  # noqa
from embedding_studio.schemas.fine_tuning import FineTuningStatus
from embedding_studio.workers.fine_tuning.experiments.finetuning_iteration import (
    FineTuningIteration,
)
from embedding_studio.workers.fine_tuning.finetune_embedding import (
    finetune_embedding_model,
)

logger = logging.getLogger(__name__)

plugin_manager = PluginManager()
plugin_manager.discover_plugins(directory=settings.ES_PLUGINS_PATH)


class FineTuningWorkerException(Exception):
    """
    Custom exception class for fine-tuning related errors.
    """


@dramatiq.actor(
    queue_name="fine_tuning_worker",
    max_retries=settings.FINE_TUNING_WORKER_MAX_RETRIES,
    time_limit=settings.FINE_TUNING_WORKER_TIME_LIMIT,
)
def fine_tuning_worker(task_id: str):
    """
    Dramatiq task for fine-tuning a model.

    Args:
        task_id (str): The ID of the fine-tuning task.
    """
    logger.info(f"start fine_tuning_worker: task_id={task_id}")

    task = fine_tuning_task.get(id=task_id)
    if not task:
        raise FineTuningWorkerException(f"Task with ID `{task_id}` not found")
    task.status = FineTuningStatus.processing
    fine_tuning_task.update(obj=task)

    # get from clickstream service
    # TODO: replace with real data
    logger.debug(f"load clickstream")
    clicstream_example = "clickstream_example.json"
    with open(clicstream_example, "r") as f:
        clickstream = json.load(f)
    clickstream = clickstream[:100]

    try:
        fine_tuning_plugin = plugin_manager.get_plugin(task.fine_tuning_method)
        if not fine_tuning_plugin:
            raise FineTuningWorkerException(
                f"Fine tuning plugin with name `{task.fine_tuning_method}` "
                f"not found"
            )
        builder = fine_tuning_plugin.get_fine_tuning_builder(
            clickstream=clickstream
        )
        iteration = FineTuningIteration(
            start=task.start_at,
            end=task.end_at,
            plugin_name=task.fine_tuning_method,
        )
        logger.info("start finetune_embedding_model")
        finetune_embedding_model(
            iteration=iteration,
            settings=builder.fine_tuning_settings,
            ranking_data=builder.ranking_data,
            query_retriever=builder.query_retriever,
            tracker=builder.experiments_manager,
            initial_params=builder.initial_params,
            initial_max_evals=builder.initial_max_evals,
        )

    except Exception:
        try:
            task.status = FineTuningStatus.error
            fine_tuning_task.update(obj=task)
        except Exception as exc:
            logger.exception(f"Failed to update task status: {exc}")
        raise

    task.status = FineTuningStatus.done
    fine_tuning_task.update(obj=task)
