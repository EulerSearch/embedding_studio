from dramatiq import Actor

from embedding_studio.context.app_context import context
from embedding_studio.models.reindex import ReindexTaskInDb
from embedding_studio.models.task import TaskStatus
from embedding_studio.workers.upsertion.utils.reindex import (
    handle_reindex_error,
    logger,
    plugin_manager,
    process_reindex,
)


def handle_reindex(task: ReindexTaskInDb, reindex_subworker: Actor):
    """
    Handles the full reindex process for a given task.
    """
    logger.info(f"Starting reindex process for task ID: {task.id}")

    # Update task status to processing
    task.status = TaskStatus.processing
    context.reindex_task.update(obj=task)

    vector_db = context.vectordb
    plugin = plugin_manager.get_plugin(task.source.fine_tuning_method)
    embedding_model_info = plugin.get_embedding_model_info(
        task.source.embedding_model_id
    )
    dest_embedding_model_info = plugin.get_embedding_model_info(
        task.dest.embedding_model_id
    )

    # Get collections for source and destination
    source_collection = vector_db.get_collection(embedding_model_info)
    if not source_collection:
        logger.error(f"Source collection is not found [task ID: {task.id}]")
        task.status = TaskStatus.failed
        context.reindex_task.update(obj=task)

    logger.info(f"Creating Vector DB dest collection [task ID: {task.id}]")
    try:
        _ = vector_db.get_or_create_collection(
            dest_embedding_model_info, plugin.get_search_index_info()
        )  # Ensure destination collection exists

    except Exception as e:
        logger.exception(
            f"Something went wrong during "
            f"collection retrieval / creation [task ID: {task.id}]"
        )
        handle_reindex_error(task, e)

    try:
        process_reindex(task, source_collection, reindex_subworker)
    except Exception as e:
        handle_reindex_error(task, e)
