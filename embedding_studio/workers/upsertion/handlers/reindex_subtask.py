import logging
import traceback

from embedding_studio.context.app_context import context
from embedding_studio.core.config import settings
from embedding_studio.core.plugin import PluginManager
from embedding_studio.models.items_handler import DataItem
from embedding_studio.models.reindex import ReindexSubtaskInDb
from embedding_studio.models.task import TaskStatus
from embedding_studio.workers.upsertion.utils.exceptions import (
    ReindexException,
)
from embedding_studio.workers.upsertion.utils.upsert import process_upsert

logger = logging.getLogger(__name__)

# Initialize plugin manager and discover plugins
plugin_manager = PluginManager()
plugin_manager.discover_plugins(directory=settings.ES_PLUGINS_PATH)


def handle_reindex_subtask(task: ReindexSubtaskInDb):
    """
    Handles the full reindex process for a given task for a batch of data items.
    """
    logger.info(
        f"Starting reindex subprocess for task ID: {task.parent_id} [{task.offset}:{task.offset+task.limit}]"
    )

    # Update task status to processing
    task.status = TaskStatus.processing
    context.reindex_subtask.update(obj=task)

    source_iteration = context.mlflow_client.get_iteration_by_id(
        task.source.embedding_model_id
    )
    if source_iteration is None:
        task.status = TaskStatus.failed
        context.reindex_task.update(obj=task)
        raise ReindexException(
            f"Fine tuning iteration with ID"
            f"[{task.source.embedding_model_id}] does not exist."
        )

    source_plugin = plugin_manager.get_plugin(source_iteration.plugin_name)
    embedding_model_info = source_plugin.get_embedding_model_info(
        task.source.embedding_model_id
    )

    dest_iteration = context.mlflow_client.get_iteration_by_id(
        task.dest.embedding_model_id
    )
    if dest_iteration is None:
        task.status = TaskStatus.failed
        context.reindex_task.update(obj=task)
        raise ReindexException(
            f"Fine tuning iteration with ID"
            f"[{task.dest.embedding_model_id}] does not exist."
        )

    dest_plugin = plugin_manager.get_plugin(dest_iteration.plugin_name)
    dest_embedding_model_info = dest_plugin.get_embedding_model_info(
        task.dest.embedding_model_id
    )

    # Get collections for source and destination
    source_collection = context.vectordb.get_collection(
        embedding_model_info.id
    )
    if not source_collection:
        logger.error(f"Source collection is not found [task ID: {task.id}]")
        task.status = TaskStatus.failed
        context.reindex_subtask.update(obj=task)
        return

    # Get collections for source and destination
    dest_collection = context.vectordb.get_collection(
        dest_embedding_model_info.id
    )
    if not source_collection:
        logger.error(f"Dest collection is not found [task ID: {task.id}]")
        task.status = TaskStatus.failed
        context.reindex_subtask.update(obj=task)
        return

    try:
        objects_common_data_batch = (
            source_collection.get_objects_common_data_batch(
                task.limit, task.offset
            )
        )
        items = [
            DataItem(
                object_id=info.object_id,
                payload=info.payload,
                item_info=info.storage_meta,
            )
            for info in objects_common_data_batch.objects_info
        ]

        task.items = items

    except Exception:
        tb = traceback.format_exc()
        logger.exception(f"Can't get uploading items [task ID: {task.id}]")
        task.status = TaskStatus.failed
        task.detail = tb[-1500:]
        context.reindex_subtask.update(obj=task)
        return

    data_loader = dest_plugin.get_data_loader()
    items_splitter = dest_plugin.get_items_splitter()
    inference_client = dest_plugin.get_inference_client_factory().get_client(
        task.dest.embedding_model_id
    )

    process_upsert(
        task,
        dest_collection,
        data_loader,
        items_splitter,
        inference_client,
        context.reindex_subtask,
    )

    logger.info(
        f"Reindex subprocess for task ID: {task.parent_id} [{task.offset}:{task.offset+task.limit}] is finished."
    )
