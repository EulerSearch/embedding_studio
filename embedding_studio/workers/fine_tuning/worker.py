import logging
import uuid

import dramatiq

from embedding_studio.context.app_context import context
from embedding_studio.core.config import settings
from embedding_studio.core.plugin import PluginManager
from embedding_studio.db.redis import redis_broker  # noqa
from embedding_studio.experiments.finetuning_iteration import (
    FineTuningIteration,
)
from embedding_studio.models.task import TaskStatus
from embedding_studio.utils.dramatiq_middlewares import (
    ActionsOnStartMiddleware,
)
from embedding_studio.utils.initializer_actions import init_nltk
from embedding_studio.workers.fine_tuning.finetune_embedding import (
    finetune_embedding_model,
)

logger = logging.getLogger(__name__)

plugin_manager = PluginManager()
plugin_manager.discover_plugins(directory=settings.ES_PLUGINS_PATH)

redis_broker.add_middleware(ActionsOnStartMiddleware([init_nltk]))


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
    logger.info(f"Start fine-tuning worker with task ID `{task_id}`")

    task = context.fine_tuning_task.get(id=task_id)
    if not task:
        raise FineTuningWorkerException(f"Task with ID `{task_id}` not found")

    try:
        task.status = TaskStatus.processing
        context.fine_tuning_task.update(obj=task)

        if not task.batch_id:
            release_id = task.idempotency_key or task.id or uuid.uuid4()
            logger.info(f"Release batch with ID `{release_id}`")
            session_batch = context.clickstream_dao.release_batch(
                release_id=str(release_id)
            )
            if session_batch is None:
                task.status = TaskStatus.refused
                context.fine_tuning_task.update(obj=task)
                raise FineTuningWorkerException(
                    f"Cannot release batch with ID `{release_id}`"
                )
            task.batch_id = session_batch.batch_id
            context.fine_tuning_task.update(obj=task)

        # TODO: add config with parameters
        clickstream = context.clickstream_dao.get_batch_sessions(task.batch_id)
        if not clickstream:
            task.status = TaskStatus.refused
            context.fine_tuning_task.update(obj=task)

            raise FineTuningWorkerException(
                f"Clickstream batch with ID `{task.batch_id}` not found"
            )

        fine_tuning_plugin = plugin_manager.get_plugin(task.fine_tuning_method)
        if not fine_tuning_plugin:
            task.status = TaskStatus.refused
            context.fine_tuning_task.update(obj=task)

            raise FineTuningWorkerException(
                f"Fine tuning plugin with name `{task.fine_tuning_method}` "
                f"not found"
            )
        if not fine_tuning_plugin.get_manager().has_initial_model():
            logger.info("No initial model found, uploading.")
            logger.info(f"Upload initial model...")
            fine_tuning_plugin.upload_initial_model()
            logger.info(f"Upload initial model... OK")

        logger.info("Create fine-tuning builder...")
        builder = fine_tuning_plugin.get_fine_tuning_builder(
            clickstream=clickstream
        )
        logger.info("Create fine-tuning builder... OK")

        iteration = FineTuningIteration(
            batch_id=task.batch_id,
            run_id=task.embedding_model_id,
            plugin_name=task.fine_tuning_method,
        )
        logger.info("Start fine-tuning the embedding model...")
        finetune_embedding_model(
            iteration=iteration,
            settings=builder.fine_tuning_settings,
            ranking_data=builder.ranking_data,
            query_retriever=builder.query_retriever,
            tracker=builder.experiments_manager,
            initial_params=builder.initial_params,
            initial_max_evals=builder.initial_max_evals,
        )
        logger.info(
            "Fine tuning of the embedding model was completed successfully!"
        )
        builder.experiments_manager.set_iteration(iteration)
        best_run_id = builder.experiments_manager.get_best_current_run_id()
        best_model_url = builder.experiments_manager.get_current_model_url()
        logger.info(
            f"You can download best model using this url: {best_model_url}"
        )
        task.best_run_id = best_run_id
        task.best_model_url = best_model_url
        builder.experiments_manager.finish_iteration()

    except Exception:
        try:
            task.status = TaskStatus.failed
            context.fine_tuning_task.update(obj=task)
        except Exception as exc:
            logger.exception(f"Failed to update task status: {exc}")
        raise

    task.status = TaskStatus.done
    context.fine_tuning_task.update(obj=task)
