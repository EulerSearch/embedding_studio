import fcntl
import gc
import logging
import os
import shutil

import dramatiq
import torch.cuda
from dramatiq.middleware import Middleware

from embedding_studio.context.app_context import context
from embedding_studio.core.config import settings
from embedding_studio.core.plugin import PluginManager
from embedding_studio.db.redis import redis_broker
from embedding_studio.models.inference_deployment import DeploymentStatus
from embedding_studio.workers.inference.utils.prepare_for_triton import (
    convert_for_triton,
)

# Set up logging
logger = logging.getLogger(__name__)

# Initialize and discover plugins
plugin_manager = PluginManager()
plugin_manager.discover_plugins(directory=settings.ES_PLUGINS_PATH)


def acquire_lock(lock_file_path):
    """
    Acquire an exclusive lock on the given file.
    """
    lock_file = open(lock_file_path, "w")
    fcntl.lockf(lock_file, fcntl.LOCK_EX)
    return lock_file


def release_lock(lock_file):
    """
    Release the lock on the given file.
    """
    fcntl.lockf(lock_file, fcntl.LOCK_UN)
    lock_file.close()


def init_model_repo_for_plugin(model_repo: str, plugin_name: str):
    """
    Initialize the model repository for a specific plugin.
    Uploads initial models or converts best models for Triton if necessary.
    """
    plugin = plugin_manager.get_plugin(plugin_name)
    if not plugin.manager.has_initial_model():
        logger.info("No initial model found, uploading")
        plugin.upload_initial_model()

    last_finished_iteration = plugin.manager.get_last_finished_iteration_id()
    experiment_id = (
        "initial_model"
        if last_finished_iteration is None
        else last_finished_iteration
    )
    model = (
        plugin.manager.download_initial_model()
        if last_finished_iteration is None
        else plugin.manager.download_best_model(last_finished_iteration)
    )
    convert_for_triton(model, plugin_name, model_repo, 1, experiment_id)
    del model
    gc.collect()
    torch.cuda.empty_cache()


class OnStartMiddleware(Middleware):
    """
    Middleware to handle worker start events.
    Ensures all plugins have their models initialized in the model repository.
    """

    def after_worker_boot(self, broker, worker):
        super().after_worker_boot(broker, worker)
        model_repo = os.getenv("MODEL_REPOSITORY", os.getcwd())
        for plugin_name in settings.INFERENCE_USED_PLUGINS:
            query_path = os.path.join(model_repo, plugin_name + "_query")
            items_path = os.path.join(model_repo, plugin_name + "_items")

            lock_file_path = f"/tmp/deployment_lock_{plugin_name}.lock"
            lock_file = acquire_lock(lock_file_path)
            try:
                if not os.path.exists(
                    os.path.join(query_path, "1", "model.pt")
                ) or not (
                    os.path.exists(os.path.join(items_path, "1", "model.pt"))
                    or os.path.exists(
                        os.path.join(query_path, "1", "same_query")
                    )
                ):
                    logger.info(f"No model was uploaded for {plugin_name}")
                    init_model_repo_for_plugin(model_repo, plugin_name)
            finally:
                release_lock(lock_file)

        with open(
            os.path.join(model_repo, "initialization_complete.flag"), "w"
        ) as f:
            f.write("TRUE")


redis_broker.add_middleware(OnStartMiddleware())


class InferenceWorkerException(Exception):
    """
    Custom exception class for inference-related errors.
    """


# Actor for handling green deployment of models
@dramatiq.actor(
    queue_name="deployment_worker",
    max_retries=settings.FINE_TUNING_WORKER_MAX_RETRIES,
    time_limit=settings.FINE_TUNING_WORKER_TIME_LIMIT,
)
def deployment_worker(task_id: str):
    handle_deployment(task_id)


def handle_deployment(task_id: str):
    """
    Generalized function to handle different stages of model deployment.
    Adjusts model repository based on the deployment stage: green, blue, or revert.
    """
    model_repo = os.getenv("MODEL_REPOSITORY", os.getcwd())
    task = context.deployment_task.get(id=task_id)

    if not task:
        raise InferenceWorkerException(f"Task with ID `{task_id}` not found")

    if task.fine_tuning_method not in settings.INFERENCE_USED_PLUGINS:
        raise InferenceWorkerException(
            f'Passed plugin is not in the used plugin list ({", ".join(settings.INFERENCE_USED_PLUGINS)}).'
        )

    query_path = os.path.join(model_repo, task.fine_tuning_method + "_query")
    items_path = os.path.join(model_repo, task.fine_tuning_method + "_items")

    def manage_path(path, stage):
        if stage == "green":
            if not os.path.exists(os.path.join(path, "1", "model.pt")):
                logger.info(
                    f"Initializing {task.fine_tuning_method} for green deployment at {path}."
                )
                init_model_repo_for_plugin(model_repo, task.fine_tuning_method)
            else:
                logger.info(
                    f"Updating {task.fine_tuning_method} to new version for green deployment at {path}."
                )
                if os.path.exists(os.path.join(path, "2")):
                    shutil.rmtree(os.path.join(path, "2"))
                shutil.copytree(
                    os.path.join(path, "1"), os.path.join(path, "2")
                )

        elif stage == "blue":
            if os.path.exists(os.path.join(path, "2", "model.pt")):
                logger.info(f"Promoting green to blue deployment at {path}.")
                if os.path.exists(os.path.join(path, "1")):
                    shutil.rmtree(os.path.join(path, "1"))
                shutil.move(os.path.join(path, "2"), os.path.join(path, "1"))
            else:
                logger.error(
                    f"No green model available for promotion to blue at {path}."
                )

        elif stage == "revert":
            archived_path = os.path.join(path, "_archived")
            if os.path.exists(os.path.join(archived_path, "model.pt")):
                logger.info(f"Reverting to archived blue model at {path}.")
                if os.path.exists(os.path.join(path, "1")):
                    shutil.rmtree(os.path.join(path, "1"))
                shutil.move(archived_path, os.path.join(path, "1"))
            else:
                logger.error(
                    f"No archived model available to revert at {path}."
                )

    lock_file_path = f"/tmp/deployment_lock_{task_id}.lock"
    lock_file = acquire_lock(lock_file_path)
    try:
        task.status = DeploymentStatus.processing
        context.deployment_task.update(obj=task)

        manage_path(query_path, task.stage)
        manage_path(items_path, task.stage)
    except Exception:
        task.status = DeploymentStatus.error
        context.deployment_task.update(obj=task)

    finally:
        task.status = DeploymentStatus.done
        context.deployment_task.update(obj=task)
        release_lock(lock_file)
