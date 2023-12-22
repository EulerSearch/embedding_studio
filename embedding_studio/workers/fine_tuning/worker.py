import logging
import uuid

import dramatiq

from embedding_studio.context.app_context import context
from embedding_studio.core.config import settings
from embedding_studio.core.plugin import PluginManager
from embedding_studio.db.redis import redis_broker  # noqa
from embedding_studio.models.fine_tuning import FineTuningStatus
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
    """Dramatiq task for fine-tuning a model.

    :param task_id: The ID of the fine-tuning task.
    """
    logger.info(f"start fine_tuning_worker: task_id={task_id}")

    task = context.fine_tuning_task.get(id=task_id)
    if not task:
        raise FineTuningWorkerException(f"Task with ID `{task_id}` not found")

    try:
        task.status = FineTuningStatus.processing
        context.fine_tuning_task.update(obj=task)

        if not task.batch_id:
            release_id = uuid.uuid4()
            logger.info(f"create release with ID `{release_id}`")
            session_batch = context.clickstream_dao.release_batch(
                release_id=str(release_id)
            )
            if session_batch is None:
                raise FineTuningWorkerException(
                    f"Cannot release batch with ID `{release_id}`"
                )
            task.batch_id = session_batch.batch_id
            context.fine_tuning_task.update(obj=task)

        # TODO: add config with parameters
        clickstream = context.clickstream_dao.get_batch_sessions(task.batch_id)
        if not clickstream:
            raise FineTuningWorkerException(
                f"Clickstream batch with ID `{task.batch_id}` not found"
            )

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
            batch_id=task.batch_id,
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

        best_model_url = builder.experiments_manager.get_last_model_url()
        logger.info(
            f"You can download best model using this url: {best_model_url}"
        )
        task.best_model_url = best_model_url

    except Exception:
        try:
            task.status = FineTuningStatus.error
            context.fine_tuning_task.update(obj=task)
        except Exception as exc:
            logger.exception(f"Failed to update task status: {exc}")
        raise

    task.status = FineTuningStatus.done
    context.fine_tuning_task.update(obj=task)
