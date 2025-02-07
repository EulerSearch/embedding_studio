from dramatiq import Actor

from embedding_studio.context.app_context import context
from embedding_studio.core.config import settings
from embedding_studio.models.reindex import ReindexTaskInDb
from embedding_studio.models.task import TaskStatus
from embedding_studio.utils.plugin_utils import get_vectordb
from embedding_studio.workers.upsertion.utils.deployment import (
    blue_switch,
    initiate_model_deployment_and_wait,
)
from embedding_studio.workers.upsertion.utils.exceptions import (
    ReindexException,
)
from embedding_studio.workers.upsertion.utils.reindex import (
    handle_reindex_error,
    logger,
    plugin_manager,
    process_reindex,
)


def handle_reindex(
    task: ReindexTaskInDb,
    reindex_subworker: Actor,
    deployment_worker: Actor,
    deletion_worker: Actor,
):
    """
    Handles the full reindex process for a given task.
    """
    logger.info(f"Starting reindex process for task ID: {task.id}")

    # Update task status to processing
    task.status = TaskStatus.processing
    context.reindex_task.update(obj=task)

    source_iteration = context.mlflow_client.get_iteration_by_id(
        task.source.embedding_model_id
    )
    if source_iteration is None:
        task.status = TaskStatus.failed
        context.deletion_task.update(obj=task)
        raise ReindexException(
            f"Fine tuning iteration with ID"
            f"[{task.source.embedding_model_id}] does not exist."
        )

    source_plugin = plugin_manager.get_plugin(source_iteration.plugin_name)
    source_vectordb = get_vectordb(source_plugin)

    embedding_model_info = source_plugin.get_embedding_model_info(
        task.source.embedding_model_id
    )

    dest_iteration = context.mlflow_client.get_iteration_by_id(
        task.dest.embedding_model_id
    )
    if dest_iteration is None:
        task.status = TaskStatus.failed
        context.deletion_task.update(obj=task)
        raise ReindexException(
            f"Fine tuning iteration with ID"
            f"[{task.dest.embedding_model_id}] does not exist."
        )

    dest_plugin = plugin_manager.get_plugin(dest_iteration.plugin_name)
    dest_vectordb = get_vectordb(dest_plugin)

    dest_embedding_model_info = dest_plugin.get_embedding_model_info(
        task.dest.embedding_model_id
    )

    # Get collections for source and destination
    source_collection = source_vectordb.get_collection(embedding_model_info.id)
    if not source_collection:
        logger.error(f"Source collection is not found [task ID: {task.id}]")
        task.status = TaskStatus.failed
        context.reindex_task.update(obj=task)

    logger.info(f"Creating Vector DB dest collection [task ID: {task.id}]")
    try:

        _ = dest_vectordb.get_or_create_collection(
            dest_embedding_model_info
        )  # Ensure destination collection exists

        _ = dest_vectordb.get_or_create_query_collection(
            dest_embedding_model_info
        )

    except Exception as e:
        logger.exception(
            f"Something went wrong during "
            f"collection retrieval / creation [task ID: {task.id}]"
        )
        handle_reindex_error(task, e)
        return

    inference_client = dest_plugin.get_inference_client_factory().get_client(
        task.dest.embedding_model_id
    )

    try:
        if not inference_client.is_model_ready():
            logger.warning(
                f"Dest model with ID {task.dest.embedding_model_id} is not deployed."
            )
            if settings.REINDEX_INITIATE_MODEL_DEPLOYMENT:
                logger.info(
                    f"Starting deployment for model with ID {task.dest.embedding_model_id}"
                )
                initiate_model_deployment_and_wait(
                    dest_plugin, task, task.dest, deployment_worker
                )

            else:
                logger.error("Stop reindex execution.")
                raise KeyError(
                    f"Model with ID {task.dest.embedding_model_id} not found."
                )

    except Exception as e:
        logger.exception(
            f"Something went wrong during "
            f"model creation [task ID: {task.id}]"
        )
        handle_reindex_error(task, e)
        return

    try:
        process_reindex(task, source_collection, reindex_subworker)

        logger.info("Reindexing complete.")
        if task.deploy_as_blue:
            blue_switch(task, deletion_worker=deletion_worker)

    except Exception as e:
        logger.exception(
            f"Something went wrong during reindexing [task ID: {task.id}]"
        )
        handle_reindex_error(task, e)
