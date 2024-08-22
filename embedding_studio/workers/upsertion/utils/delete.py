import logging
import traceback

from embedding_studio.context.app_context import context
from embedding_studio.core.config import settings
from embedding_studio.core.plugin import PluginManager
from embedding_studio.models.delete import DeletionTaskInDb
from embedding_studio.models.task import TaskStatus
from embedding_studio.models.utils import create_failed_deletion_data_item
from embedding_studio.workers.upsertion.utils.collection import get_collection

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

    vector_db = context.vectordb
    plugin = plugin_manager.get_plugin(task.fine_tuning_method)

    collection = get_collection(vector_db, plugin, task)
    if not collection:
        logger.error(
            f"Task {task.id} is finished with error: collection not found."
        )
        task.status = TaskStatus.failed
        context.deletion_task.update(obj=task)
        return

    logger.info(f"Start embeddings deletion [task ID: {task.id}]")
    try:
        collection.delete(task.object_ids)

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
