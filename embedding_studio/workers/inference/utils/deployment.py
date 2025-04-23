import logging
import os
import tempfile

from embedding_studio.context.app_context import context
from embedding_studio.core.config import settings
from embedding_studio.models.task import TaskStatus
from embedding_studio.workers.inference.utils.exceptions import (
    InferenceWorkerException,
)
from embedding_studio.workers.inference.utils.file_locks import (
    acquire_lock,
    release_lock,
)
from embedding_studio.workers.inference.utils.prepare_for_triton import (
    convert_for_triton,
)

logger = logging.getLogger(__name__)


def handle_deployment(task_id: str):
    """
    Generalized function to handle embedding model deployment to Triton Inference Server.

    This function manages the entire deployment workflow:
    1. Retrieves the deployment task by ID
    2. Validates the embedding model exists and uses a supported plugin
    3. Ensures deployment limits haven't been exceeded
    4. Downloads the model from MLflow
    5. Converts and deploys the model to Triton Inference Server

    The function uses file locking to prevent concurrent deployments of the same model,
    which could lead to race conditions or corrupted model files.
    """

    def handle_deployment(task_id: str):
        """
        Handles the deployment of an embedding model to Triton Inference Server.

        This includes:
        - Retrieving and validating the deployment task
        - Ensuring the model exists and uses a supported plugin
        - Enforcing the maximum deployed model limit
        - Downloading the model from MLflow
        - Converting and deploying it to the Triton model repository
        - Managing deployment status and locking to prevent race conditions

        :param task_id: ID of the model deployment task to process
        """
        model_repo = os.getenv("MODEL_REPOSITORY", os.getcwd())

        # Fetch deployment task from context
        task = context.model_deployment_task.get(id=task_id)
        if not task:
            raise InferenceWorkerException(
                f"Deployment task with ID `{task_id}` not found"
            )

        # Fetch model iteration from MLflow
        iteration = context.mlflow_client.get_iteration_by_id(
            task.embedding_model_id
        )
        if iteration is None:
            task.status = TaskStatus.failed
            context.model_deployment_task.update(obj=task)
            message = f"Can not find iteration for embedding model {task.embedding_model_id}"
            logger.error(message)
            raise InferenceWorkerException(message)

        # Check if plugin is supported for inference
        if iteration.plugin_name not in settings.INFERENCE_USED_PLUGINS:
            task.status = TaskStatus.refused
            context.model_deletion_task.update(obj=task)
            raise InferenceWorkerException(
                f"Passed plugin is not in the used plugin list"
                f' ({", ".join(settings.INFERENCE_USED_PLUGINS)}).'
            )

        # Enforce deployment limit based on number of unique deployed models
        deployed_models_list = os.listdir(model_repo)
        deployed_models_unique = set(
            tuple(deployed_model.split("_")[:-1])
            for deployed_model in deployed_models_list
        )
        if (
            len(deployed_models_unique)
            > settings.INFERENCE_WORKER_MAX_DEPLOYED_MODELS
        ):
            task.status = TaskStatus.refused
            task.detail = (
                f"Number of deployed models {len(deployed_models_unique)} "
                f"exceeds max deployed models {settings.INFERENCE_WORKER_MAX_DEPLOYED_MODELS}"
            )
            context.model_deployment_task.update(obj=task)
            return

        # Acquire file lock to prevent concurrent deployment of the same model
        temp_dir = tempfile.gettempdir()
        lock_file_path = os.path.join(
            temp_dir, f"deployment_lock_{task.embedding_model_id}.lock"
        )
        lock_file = acquire_lock(lock_file_path)
        try:
            task.status = TaskStatus.processing
            context.model_deployment_task.update(obj=task)

            # Download model using the appropriate plugin
            plugin = context.plugin_manager.get_plugin(iteration.plugin_name)
            experiments_manager = plugin.get_manager()
            model = experiments_manager.download_model_by_run_id(
                task.embedding_model_id
            )

            # Convert and deploy model to Triton
            convert_for_triton(
                model=model,
                plugin_name=iteration.plugin_name,
                model_repo=model_repo,
                model_version=1,
                embedding_model_id=task.embedding_model_id,
            )

            task.status = TaskStatus.done
            context.model_deployment_task.update(obj=task)

        except Exception:
            # In case of failure, mark task as failed
            task.status = TaskStatus.failed
            context.model_deployment_task.update(obj=task)

        finally:
            # Always release the lock
            release_lock(lock_file)
