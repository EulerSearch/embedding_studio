import logging
import traceback

from embedding_studio.context.app_context import context
from embedding_studio.core.config import settings
from embedding_studio.core.plugin import PluginManager
from embedding_studio.models.delete import DeletionTaskInDb
from embedding_studio.models.task import TaskStatus
from embedding_studio.models.utils import create_failed_deletion_data_item
from embedding_studio.utils.plugin_utils import get_vectordb

logger = logging.getLogger(__name__)

plugin_manager = PluginManager()
# Initialize and discover plugins
plugin_manager.discover_plugins(directory=settings.ES_PLUGINS_PATH)


def handle_delete(task: DeletionTaskInDb):
    """
    Handles the deletion process for a given task.

    :param task: The deletion task object in the database.
    """
    logger.info(f"Starting delete process for task ID: {task.id}")

    task.status = TaskStatus.processing
    context.deletion_task.update(obj=task)

    plugin = plugin_manager.get_plugin(task.fine_tuning_method)
    vector_db = get_vectordb(plugin)
    vector_db.update_info()

    embedding_model_info = plugin.get_embedding_model_info(
        task.embedding_model_id
    )

    logger.info(
        f"Creating or retrieving " f"Vector DB collection [task ID: {task.id}]"
    )
    try:
        collection = vector_db.get_or_create_collection(
            embedding_model_info, plugin.get_search_index_info()
        )
    except Exception:
        logger.exception(
            f"Something went wrong during "
            f"collection retrieval / creation [task ID: {task.id}]"
        )
        task.status = TaskStatus.failed
        context.deletion_task.update(obj=task)
        return

    logger.info(f"Start embeddings deletion [task ID: {task.id}]")
    try:
        objects = collection.find_by_ids(task.object_ids)
        original_ids = set()
        for obj in objects:
            if obj.original_id:
                original_ids.add(obj.original_id)

        logger.info(f"Found {len(original_ids)} original objects mentioned")

        not_original_objects = collection.find_by_original_ids(task.object_ids)
        not_original_object_ids = set()
        for obj in not_original_objects:
            not_original_object_ids.add(obj.object_id)

        logger.info(
            f"Found {len(not_original_object_ids)} changed objects mentioned"
        )

        collection.delete(
            list(
                set(
                    task.object_ids
                    + list(original_ids)
                    + list(not_original_object_ids)
                )
            )
        )

        logger.info(f"Task {task.id} is finished.")
        task.status = TaskStatus.done
        context.deletion_task.update(obj=task)

    except Exception:
        tb = traceback.format_exc()
        message = (
            f"Something went wrong during deletion "
            f"for {len(task.object_ids)} items [task ID: {task.id}]"
        )
        logger.exception(message)

        for object_id in task.object_ids:
            task.failed_item_ids.append(
                create_failed_deletion_data_item(object_id, tb)
            )

        task.status = TaskStatus.failed
        context.deletion_task.update(obj=task)
