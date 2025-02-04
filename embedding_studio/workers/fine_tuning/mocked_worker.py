import logging

import dramatiq

from embedding_studio.context.app_context import context
from embedding_studio.core.config import settings
from embedding_studio.core.plugin import PluginManager
from embedding_studio.db.redis import redis_broker  # noqa
from embedding_studio.experiments.finetuning_iteration import (
    FineTuningIteration,
)
from embedding_studio.experiments.finetuning_params import FineTuningParams
from embedding_studio.experiments.metrics_accumulator import MetricValue
from embedding_studio.models.reindex import ReindexTaskCreateSchema
from embedding_studio.models.task import ModelParams, TaskStatus
from embedding_studio.utils.dramatiq_middlewares import (
    ActionsOnStartMiddleware,
)
from embedding_studio.utils.dramatiq_task_handler import create_and_send_task
from embedding_studio.utils.initializer_actions import init_nltk
from embedding_studio.utils.plugin_utils import is_basic_plugin
from embedding_studio.workers.fine_tuning.exceptions import (
    FineTuningWorkerException,
)
from embedding_studio.workers.upsertion.worker import reindex_worker

logger = logging.getLogger(__name__)

plugin_manager = PluginManager()
plugin_manager.discover_plugins(directory=settings.ES_PLUGINS_PATH)

redis_broker.add_middleware(ActionsOnStartMiddleware([init_nltk]))


@dramatiq.actor(
    queue_name="fine_tuning_mocked_worker",
    max_retries=settings.FINE_TUNING_WORKER_MAX_RETRIES,
    time_limit=settings.FINE_TUNING_WORKER_TIME_LIMIT,
)
def fine_tuning_mocked_worker(task_id: str):
    """Simulation of a dramatiq task for fine-tuning a model.
    It will load one model only.

    :param task_id: The ID of the fine-tuning task.
    """
    logger.info(
        f"Start fine-tuning similation worker with task ID `{task_id}`"
    )

    task = context.fine_tuning_task.get(id=task_id)
    if not task:
        raise FineTuningWorkerException(f"Task with ID `{task_id}` not found")

    try:
        task.status = TaskStatus.processing
        context.fine_tuning_task.update(obj=task)

        fine_tuning_plugin = plugin_manager.get_plugin(task.fine_tuning_method)
        if not fine_tuning_plugin:
            raise FineTuningWorkerException(
                f"Fine tuning plugin with name `{task.fine_tuning_method}` "
                f"not found"
            )

        if not is_basic_plugin(fine_tuning_plugin):
            task.status = TaskStatus.refused
            context.fine_tuning_task.update(obj=task)

            raise FineTuningWorkerException(
                f"Fine tuning is not available for `{task.fine_tuning_method}` "
                f"with {fine_tuning_plugin.meta.use_case.name} use case."
            )

        if not fine_tuning_plugin.get_manager().has_initial_model():
            logger.info("No initial model found, uploading.")
            logger.info(f"Upload initial model...")
            fine_tuning_plugin.upload_initial_model()
            logger.info(f"Upload initial model... OK")

        iteration = FineTuningIteration(
            batch_id="mocked-batch",
            run_id=fine_tuning_plugin.get_manager().get_initial_run_id(),
            plugin_name="DefaultFineTuningMethod",
        )

        manager = fine_tuning_plugin.get_manager()
        manager.set_iteration(iteration)
        params = FineTuningParams(
            num_fixed_layers=6,
            query_lr=0.1,
            items_lr=0.1,
            query_weight_decay=0.1,
            items_weight_decay=0.1,
            margin=0.1,
            not_irrelevant_only=True,
            negative_downsampling=0.5,
        )
        _ = manager.set_run(params)
        manager.save_metric(
            MetricValue("not_irrelevant_dist_shift", 0.1).add_prefix("test")
        )

        inital_model = manager.download_initial_model()
        manager.save_model(inital_model, True)
        manager.finish_run()

        best_run_id, _ = manager.get_best_current_run_id()
        best_model_url = manager.get_current_model_url()

        task.best_run_id = best_run_id
        task.best_model_url = best_model_url

        manager.finish_iteration()

    except Exception:
        try:
            task.status = TaskStatus.failed
            context.fine_tuning_task.update(obj=task)

        except Exception as exc:
            logger.exception(f"Failed to update task status: {exc}")
        raise

    task.status = TaskStatus.done
    context.fine_tuning_task.update(obj=task)

    if task.deploy_as_blue:
        logger.info(
            f"Starting reindex task for model with "
            f"ID {task.fine_tuning_method}/{task.best_run_id}"
        )
        reindex_task = context.reindex_task.create(
            schema=ReindexTaskCreateSchema(
                source=ModelParams(
                    embedding_model_id=task.embedding_model_id,
                    fine_tuning_method=task.fine_tuning_method,
                ),
                dest=ModelParams(
                    embedding_model_id=task.best_run_id,
                    fine_tuning_method=task.fine_tuning_method,
                ),
                deploy_as_blue=True,
                wait_on_conflict=task.wait_on_conflict,
                parent_id=task.id,
            ),
            return_obj=True,
        )
        reindex_task = create_and_send_task(
            reindex_worker, reindex_task, context.reindex_task
        )

        if not reindex_task:
            raise Exception(
                f"Failed to create and send deployment task for model {task.best_run_id}"
            )
