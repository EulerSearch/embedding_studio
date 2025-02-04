import time
from datetime import datetime

from dramatiq import Actor

from embedding_studio.context.app_context import context
from embedding_studio.core.config import settings
from embedding_studio.core.plugin import FineTuningMethod
from embedding_studio.models.embeddings.models import EmbeddingModelInfo
from embedding_studio.models.inference_deployment_tasks import (
    ModelManagementTaskCreateSchema,
)
from embedding_studio.models.reindex import ReindexTaskInDb
from embedding_studio.models.task import ModelParams, TaskStatus
from embedding_studio.utils.dramatiq_task_handler import create_and_send_task
from embedding_studio.utils.plugin_utils import is_basic_plugin
from embedding_studio.workers.upsertion.utils.reindex import logger


def initiate_model_deployment_and_wait(
    plugin: FineTuningMethod,
    task: ReindexTaskInDb,
    embedding_model: ModelParams,
    deployment_worker: Actor,
):
    """
    Initiates model deployment and waits for it to be ready.
    """
    logger.info(
        f"Starting deployment for model with "
        f"ID {embedding_model.fine_tuning_method}/{embedding_model.embedding_model_id}"
    )
    deployment_task = context.model_deployment_task.create(
        schema=ModelManagementTaskCreateSchema(
            embedding_model_id=embedding_model.embedding_model_id,
            fine_tuning_method=embedding_model.fine_tuning_method,
            parent_id=task.id,
        ),
        return_obj=True,
    )
    # Use create_and_send_task instead of manual sending and updating
    updated_task = create_and_send_task(
        deployment_worker, deployment_task, context.model_deployment_task
    )

    if not updated_task:
        raise Exception(
            f"Failed to create and send deployment task for model {embedding_model.embedding_model_id}"
        )

    start = datetime.now()
    in_pending = True

    while True:
        updated_task = context.model_deployment_task.get(updated_task.id)
        attempt_end = datetime.now()

        if not updated_task:
            raise Exception(
                f"Failed to retrieve deployment task for model {embedding_model.embedding_model_id}"
            )
        elif updated_task.status == TaskStatus.pending:
            if (
                (attempt_end - start).seconds
                > settings.REINDEX_INITIATE_MODEL_DEPLOYMENT_PENDING_TIME
            ):
                raise TimeoutError("Deployment task is pending too long.")
            time.sleep(
                settings.REINDEX_INITIATE_MODEL_DEPLOYMENT_LOOP_WAIT_TIME
            )
        elif updated_task.status == TaskStatus.processing:
            if in_pending:
                start = datetime.now()
                in_pending = False
            if (
                attempt_end - start
            ).seconds > settings.INFERENCE_WORKER_TIME_LIMIT:
                raise TimeoutError("Deployment task is processing too long.")
            time.sleep(
                settings.REINDEX_INITIATE_MODEL_DEPLOYMENT_LOOP_WAIT_TIME
            )
        elif updated_task.status in [TaskStatus.failed, TaskStatus.refused]:
            raise Exception(f"Deployment task failed: {updated_task.detail}")
        elif updated_task.status == TaskStatus.done:
            inference_client = (
                plugin.get_inference_client_factory().get_client(
                    embedding_model.embedding_model_id
                )
            )
            if not inference_client.is_model_ready():
                if (
                    attempt_end - start
                ).seconds > settings.INFERENCE_WORKER_TIME_LIMIT:
                    raise TimeoutError("Model is not ready.")
                time.sleep(
                    settings.REINDEX_INITIATE_MODEL_DEPLOYMENT_LOOP_WAIT_TIME
                )
            else:
                break

        else:
            break


def initiate_model_deletion(
    task: ReindexTaskInDb,
    embedding_model_info: EmbeddingModelInfo,
    deletion_worker: Actor,
):
    """
    Initiates the deletion of a model.
    """
    logger.info(
        f"Initiating deletion for model with ID {embedding_model_info.id}"
    )

    deletion_task = context.model_deletion_task.create(
        schema=ModelManagementTaskCreateSchema(
            embedding_model_id=embedding_model_info.id,
            fine_tuning_method=embedding_model_info.name,
            parent_id=task.id,
        ),
        return_obj=True,
    )

    updated_task = create_and_send_task(
        deletion_worker, deletion_task, context.model_deletion_task
    )

    if not updated_task:
        raise Exception(
            f"Failed to create and send deletion task for model {embedding_model_info.id}"
        )


def blue_switch(task: ReindexTaskInDb, deletion_worker: Actor):
    """
    Handles the blue switch process by setting the destination model as blue and
    deleting the current blue model's collection.
    """
    logger.info(
        f"Switching model with ID {task.dest.embedding_model_id} to blue status."
    )

    dest_plugin = context.plugin_manager.get_plugin(
        task.dest.fine_tuning_method
    )

    embedding_model_info = EmbeddingModelInfo(
        name=task.dest.fine_tuning_method,
        id=task.dest.embedding_model_id,
        use_case=dest_plugin.meta.use_case,
    )
    if is_basic_plugin(dest_plugin):
        # Get the current blue collection and its state
        context.vectordb.set_blue_collection(embedding_model_info)
    else:
        context.categories_vectordb.set_blue_collection(embedding_model_info)

    logger.info(
        f"Deleting source model collection with ID {task.source.embedding_model_id}"
    )

    source_plugin = context.plugin_manager.get_plugin(
        task.source.fine_tuning_method
    )
    source_model_info = EmbeddingModelInfo(
        name=task.source.fine_tuning_method,
        id=task.source.embedding_model_id,
        use_case=source_plugin.meta.use_case,
    )

    context.vectordb.delete_collection(source_model_info)
    if is_basic_plugin(source_plugin):
        context.vectordb.delete_query_collection(source_model_info)
    else:
        context.categories_vectordb.delete_query_collection(source_model_info)

    logger.info(
        f"Initiating deletion of source model with ID {task.source.embedding_model_id}"
    )
    initiate_model_deletion(task, source_model_info, deletion_worker)
