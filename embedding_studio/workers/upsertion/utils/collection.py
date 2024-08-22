import logging
from typing import Union

from embedding_studio.context.app_context import context
from embedding_studio.core.plugin import FineTuningMethod
from embedding_studio.models.delete import DeletionTaskInDb
from embedding_studio.models.task import TaskStatus
from embedding_studio.models.upsert import UpsertionTaskInDb
from embedding_studio.vectordb.collection import Collection
from embedding_studio.vectordb.vectordb import VectorDb

logger = logging.getLogger(__name__)


def get_collection(
    vector_db: VectorDb,
    plugin: FineTuningMethod,
    task: Union[DeletionTaskInDb, UpsertionTaskInDb],
) -> Union[Collection, None]:
    """
    Retrieves or creates a collection in the vector database.

    :param vector_db: VectorDb instance to interact with the database.
    :param plugin: FineTuningMethod instance for plugin operations.
    :param task: The deletion or upsertion task object in the database.
    :return: Collection instance or None if retrieval/creation fails.
    """
    try:
        embedding_model_info = plugin.get_embedding_model_info(
            task.embedding_model_id
        )

        logger.info(
            f"Creating or retrieving "
            f"Vector DB collection [task ID: {task.id}]"
        )
        if not vector_db.collection_exists(embedding_model_info):
            logger.warning(
                f"Collection with name: {embedding_model_info.full_name} "
                f"does not exist [task ID: {task.id}]"
            )
            search_index_info = plugin.get_search_index_info()
            collection = vector_db.create_collection(
                embedding_model_info, search_index_info
            )
        else:
            collection = vector_db.get_collection(embedding_model_info)

        return collection

    except Exception:
        logger.exception(
            f"Something went wrong during "
            f"collection retrieval [task ID: {task.id}]"
        )
        task.status = TaskStatus.failed
        if isinstance(task, DeletionTaskInDb):
            context.deletion_task.update(obj=task)
        else:
            context.upsertion_task.update(obj=task)
        return
