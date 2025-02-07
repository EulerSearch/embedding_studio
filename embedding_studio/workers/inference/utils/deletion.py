import logging
import os
import shutil
import tempfile

from embedding_studio.context.app_context import context
from embedding_studio.core.config import settings
from embedding_studio.inference_management.triton.model_storage_info import (
    DeployedModelInfo,
    ModelStorageInfo,
)
from embedding_studio.models.task import TaskStatus
from embedding_studio.workers.inference.utils.exceptions import (
    InferenceWorkerException,
)
from embedding_studio.workers.inference.utils.file_locks import (
    acquire_lock,
    release_lock,
)

logger = logging.getLogger(__name__)


def handle_deletion(task_id: str):
    """
    Generalized function to handle model deletion.
    """
    model_repo = os.getenv("MODEL_REPOSITORY", os.getcwd())
    task = context.model_deletion_task.get(id=task_id)
    if not task:
        raise InferenceWorkerException(
            f"Deployment task with ID `{task_id}` not found"
        )

    iteration = context.mlflow_client.get_iteration_by_id(
        task.embedding_model_id
    )
    if iteration is None:
        task.status = TaskStatus.failed
        context.model_deletion_task.update(obj=task)

        message = f"Can not find iteration for embedding model {task.embedding_model_id}"
        logger.error(message)
        raise InferenceWorkerException(message)

    if iteration.plugin_name not in settings.INFERENCE_USED_PLUGINS:
        task.status = TaskStatus.refused
        context.model_deletion_task.update(obj=task)

        raise InferenceWorkerException(
            f"Passed plugin is not in the used plugin list"
            f' ({", ".join(settings.INFERENCE_USED_PLUGINS)}).'
        )

    model_id = task.embedding_model_id
    if context.reindex_locks.get_by_dst_model_id(model_id) is not None:
        task.status = TaskStatus.refused
        context.model_deletion_task.update(obj=task)

        raise InferenceWorkerException(
            f"Can not delete the model with ID [{task.embedding_model_id}]: it's being used in reindexing."
        )

    temp_dir = tempfile.gettempdir()
    lock_file_path = os.path.join(
        temp_dir, f"deployment_lock_{task.embedding_model_id}.lock"
    )
    lock_file = acquire_lock(lock_file_path)
    try:
        task.status = TaskStatus.processing
        context.model_deletion_task.update(obj=task)

        iteration = context.mlflow_client.get_iteration_by_id(
            task.embedding_model_id
        )
        if iteration is None:
            task.status = TaskStatus.failed
            logger.error(
                f"Can not find iteration for embedding model {task.embedding_model_id}"
            )
            return

        query_model_storage_info = ModelStorageInfo(
            model_repo=model_repo,
            deployed_model_info=DeployedModelInfo(
                plugin_name=iteration.plugin_name,
                model_type="query",
                embedding_model_id=task.embedding_model_id,
                version="1",
            ),
        )

        same_query = os.path.exists(
            os.path.join(query_model_storage_info.model_path, "same_query")
        )
        shutil.rmtree(query_model_storage_info.model_path)

        if not same_query:
            items_model_storage_info = ModelStorageInfo(
                model_repo=model_repo,
                deployed_model_info=DeployedModelInfo(
                    plugin_name=iteration.plugin_name,
                    model_type="items",
                    embedding_model_id=task.embedding_model_id,
                    version="1",
                ),
            )
            shutil.rmtree(items_model_storage_info.model_path)

        task.status = TaskStatus.done
        context.model_deletion_task.update(obj=task)

    except Exception:
        logger.exception(
            f"Something went wrong durong model deletion with ID: {model_id}"
        )
        task.status = TaskStatus.failed
        context.model_deletion_task.update(obj=task)

    finally:
        release_lock(lock_file)
