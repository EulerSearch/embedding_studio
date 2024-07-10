import os
import tempfile

from embedding_studio.context.app_context import context
from embedding_studio.core.config import settings
from embedding_studio.models.inference_deployment_tasks import (
    ModelDeploymentStatus,
)
from embedding_studio.workers.inference.utils.exceptions import (
    InferenceWorkerException,
)
from embedding_studio.workers.inference.utils.file_locks import (
    acquire_lock,
    release_lock,
)
from embedding_studio.workers.inference.utils.init_model_repo import (
    plugin_manager,
)
from embedding_studio.workers.inference.utils.prepare_for_triton import (
    convert_for_triton,
)


def handle_deployment(task_id: str):
    """
    Generalized function to handle model deployment.
    """
    model_repo = os.getenv("MODEL_REPOSITORY", os.getcwd())
    task = context.deployment_task.get(id=task_id)

    if not task:
        raise InferenceWorkerException(
            f"Deployment task with ID `{task_id}` not found"
        )

    if task.fine_tuning_method not in settings.INFERENCE_USED_PLUGINS:
        raise InferenceWorkerException(
            f'Passed plugin is not in the used plugin list ({", ".join(settings.INFERENCE_USED_PLUGINS)}).'
        )
    temp_dir = tempfile.gettempdir()
    lock_file_path = os.path.join(
        temp_dir, f"deployment_lock_{task.embedding_model_id}.lock"
    )
    lock_file = acquire_lock(lock_file_path)
    try:
        task.status = ModelDeploymentStatus.processing
        context.deployment_task.update(obj=task)

        plugin = plugin_manager.get_plugin(task.fine_tuning_method)
        experiments_manager = plugin.get_manager()
        model = experiments_manager.download_model_by_run_id(
            task.embedding_model_id
        )

        convert_for_triton(
            model=model,
            plugin_name=task.fine_tuning_method,
            model_repo=model_repo,
            model_version=1,
            embedding_model_id=task.embedding_model_id,
        )

        task.status = ModelDeploymentStatus.done
        context.deployment_task.update(obj=task)

    except Exception:
        task.status = ModelDeploymentStatus.error
        context.deployment_task.update(obj=task)

    finally:
        release_lock(lock_file)
