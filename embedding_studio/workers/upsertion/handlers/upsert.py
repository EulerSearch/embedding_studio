from embedding_studio.context.app_context import context
from embedding_studio.core.config import settings
from embedding_studio.models.task import TaskStatus
from embedding_studio.models.upsert import UpsertionTaskInDb
from embedding_studio.utils.plugin_utils import get_vectordb
from embedding_studio.vectordb.exceptions import LockAcquisitionError
from embedding_studio.workers.upsertion.utils.exceptions import (
    UpsertionException,
)
from embedding_studio.workers.upsertion.utils.upsert import (
    logger,
    plugin_manager,
    process_upsert,
)


def handle_upsert(task: UpsertionTaskInDb):
    """
    Handles the upsertion process for a given task.

    :param task: The upsertion task object in the database.
    """
    logger.info(f"Starting upsert process for task ID: {task.id}")

    task.status = TaskStatus.processing
    context.upsertion_task.update(obj=task)

    iteration = context.mlflow_client.get_iteration_by_id(
        task.embedding_model_id
    )
    if iteration is None:
        task.status = TaskStatus.failed
        context.upsertion_task.update(obj=task)
        raise UpsertionException(
            f"Fine tuning iteration with ID"
            f"[{task.embedding_model_id}] does not exist."
        )

    plugin = plugin_manager.get_plugin(iteration.plugin_name)

    vector_db = get_vectordb(plugin)
    vector_db.update_info()
    embedding_model_info = plugin.get_embedding_model_info(
        task.embedding_model_id
    )

    data_loader = plugin.get_data_loader()
    items_splitter = plugin.get_items_splitter()
    preprocessor = plugin.get_items_preprocessor()
    inference_client = plugin.get_inference_client_factory().get_client(
        task.embedding_model_id
    )

    logger.info(
        f"Creating or retrieving " f"Vector DB collection [task ID: {task.id}]"
    )
    try:
        collection = vector_db.get_or_create_collection(embedding_model_info)
    except Exception:
        logger.exception(
            f"Something went wrong during "
            f"collection retrieval / creation [task ID: {task.id}]"
        )
        task.status = TaskStatus.failed
        context.upsertion_task.update(obj=task)
        return

    # Extract all object IDs from the task items
    batches = len(task.items) // settings.UPSERTION_BATCH_SIZE + 1
    logger.info(
        f"Start embeddings prediction for {batches} batches "
        f"[task ID: {task.id}]"
    )

    try:
        process_upsert(
            task,
            collection,
            data_loader,
            items_splitter,
            preprocessor,
            inference_client,
            context.upsertion_task,
        )

        logger.info(f"Task {task.id} is finished.")
        task.status = TaskStatus.done
        context.upsertion_task.update(obj=task)

    except LockAcquisitionError as e:
        logger.error(
            f"Failed to acquire lock for task {task.id}. "
            f"Aborting the task. Error: {str(e)}"
        )
        task.status = TaskStatus.failed
        context.upsertion_task.update(obj=task)
        return
