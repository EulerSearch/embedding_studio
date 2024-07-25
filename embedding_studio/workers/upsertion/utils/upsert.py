import logging
import traceback
from typing import List, Tuple, Union

from embedding_studio.context.app_context import context
from embedding_studio.core.config import settings
from embedding_studio.core.plugin import FineTuningMethod, PluginManager
from embedding_studio.data_storage.loaders.data_loader import DataLoader
from embedding_studio.embeddings.inference.triton.client import TritonClient
from embedding_studio.embeddings.splitters.item_splitter import ItemSplitter
from embedding_studio.models.upsert import (
    DataItem,
    UpsertionFailureStage,
    UpsertionStatus,
    UpsertionTaskInDb,
)
from embedding_studio.models.utils import create_failed_data_item
from embedding_studio.vectordb.collection import Collection
from embedding_studio.vectordb.vectordb import VectorDb
from embedding_studio.workers.upsertion.utils.exceptions import (
    DownloadException,
    InferenceException,
    SplitException,
    UploadException,
)
from embedding_studio.workers.upsertion.utils.upsertion_stages import (
    download_items,
    run_inference,
    split_items,
    upload_vectors,
)

logger = logging.getLogger(__name__)

plugin_manager = PluginManager()
# Initialize and discover plugins
plugin_manager.discover_plugins(directory=settings.ES_PLUGINS_PATH)


def get_collection(
    vector_db: VectorDb, plugin: FineTuningMethod, task: UpsertionTaskInDb
) -> Union[Collection, None]:
    """
    Retrieves or creates a collection in the vector database.

    :param vector_db: VectorDb instance to interact with the database.
    :param plugin: FineTuningMethod instance for plugin operations.
    :param task: The upsertion task object in the database.
    :return: Collection instance or None if retrieval/creation fails.
    """
    try:
        embedding_model_info = plugin.get_embedding_model_info(
            task.embedding_model_id
        )

        logger.info(
            f"Creating or retrieving Vector DB collection [task ID: {task.id}]"
        )
        if not vector_db.collection_exists(embedding_model_info):
            logger.warning(
                f"Collection with name: {embedding_model_info.full_name} does not exist [task ID: {task.id}]"
            )
            collection = vector_db.create_collection(embedding_model_info)
        else:
            collection = vector_db.get_collection(embedding_model_info)

        return collection

    except Exception:
        logger.exception(
            f"Something went wrong during collection retrieval [task ID: {task.id}]"
        )
        task.status = UpsertionStatus.error
        context.upsertion_task.update(obj=task)
        return


def handle_failed_items(
    failed_items: List[Tuple[DataItem, str]],
    task: UpsertionTaskInDb,
    exception: Exception,
):
    """
    Handles failed items during different stages of the upsertion process.

    :param failed_items: List of tuples containing failed DataItem and traceback.
    :param task: The upsertion task object in the database.
    :param exception: The exception that occurred.
    """
    stage = None
    stage_description = ""
    if isinstance(exception, DownloadException):
        stage = UpsertionFailureStage.on_downloading
        stage_description = "downloading items batch"

    elif isinstance(exception, SplitException):
        stage = UpsertionFailureStage.on_splitting
        stage_description = "splitting"

    elif isinstance(exception, InferenceException):
        stage = UpsertionFailureStage.on_inference
        stage_description = "running inference"

    elif isinstance(exception, UploadException):
        stage = UpsertionFailureStage.on_upsert
        stage_description = "uploading vectors in DB"

    message = f"Something went wrong during {stage_description} for {len(failed_items)} items [task ID: {task.id}]"
    logger.exception(message)

    for item, tb in failed_items:
        task.failed_items.append(create_failed_data_item(item, tb, stage))

    context.upsertion_task.update(obj=task)
    if not settings.UPSERTION_IGNORE_FAILED_ITEMS:
        task.status = UpsertionStatus.error
        context.upsertion_task.update(obj=task)
        raise ValueError(message)


def upsert_batch(
    batch: List[DataItem],
    data_loader: DataLoader,
    items_splitter: ItemSplitter,
    inference_client: TritonClient,
    collection: Collection,
    batch_index: int,
    task: UpsertionTaskInDb,
):
    """
    Handles the upsertion process for a single batch of items.

    :param batch: List of DataItems to be processed.
    :param data_loader: DataLoader instance to download data.
    :param items_splitter: ItemSplitter instance to split data into parts.
    :param inference_client: TritonClient instance to perform inference.
    :param collection: Collection instance to upload vectors to.
    :param batch_index: Index of the current batch.
    :param task: The upsertion task object in the database.
    """
    id_to_index = dict()
    for i, item in enumerate(batch):
        id_to_index[item.object_id] = i

    try:
        logger.info(
            f"Download items for {batch_index} batch [task ID: {task.id}]"
        )
        downloaded_items = download_items(batch, data_loader)

        logger.info(
            f"Split items data for {batch_index} batch with {len(downloaded_items)} items in it [task ID: {task.id}]"
        )
        parts, object_to_parts, failed = split_items(
            downloaded_items, items_splitter
        )
        logger.info(
            f"Split result for {batch_index} batch: {len(downloaded_items)} items -> {len(parts)} parts, [task ID: {task.id}]"
        )

        if len(failed) > 0:
            handle_failed_items(
                failed_items=[
                    (batch[id_to_index[item.meta.object_id]], tb)
                    for item, tb in failed
                ],
                task=task,
                exception=SplitException(),
            )

        logger.info(
            f"Run inference for {batch_index} batch with {len(parts)} parts in total [task ID: {task.id}]"
        )
        vectors = run_inference(parts, inference_client)

        logger.info(
            f"Upload vectors for {batch_index} batch [dims: {vectors.shape}] [task ID: {task.id}]"
        )
        upload_vectors(
            items=downloaded_items,
            vectors=vectors,
            object_to_parts=object_to_parts,
            collection=collection,
        )

    except Exception as e:
        tb = traceback.format_exc()
        handle_failed_items(
            failed_items=[(item, tb) for item in batch], task=task, exception=e
        )


def handle_upsert(task: UpsertionTaskInDb):
    """
    Handles the upsertion process for a given task.

    :param task: The upsertion task object in the database.
    """
    logger.info(f"Starting upsert process for task ID: {task.id}")

    task.status = UpsertionStatus.processing
    context.upsertion_task.update(obj=task)

    vector_db = context.vectordb
    plugin = plugin_manager.get_plugin(task.fine_tuning_method)
    data_loader = plugin.get_data_loader()
    items_splitter = plugin.get_items_splitter()
    inference_client = plugin.get_inference_client_factory().get_client(
        task.embedding_model_id
    )

    collection = get_collection(vector_db, plugin, task)
    if not collection:
        return

    batches = len(task.items) // settings.UPSERTION_BATCH_SIZE + 1
    logger.info(
        f"Start embeddings prediction for {batches} batches [task ID: {task.id}]"
    )
    for batch_index in range(batches):
        start = batch_index * settings.UPSERTION_BATCH_SIZE
        end = min(
            (batch_index + 1) * settings.UPSERTION_BATCH_SIZE,
            len(task.items),
        )

        if end <= len(task.items):
            batch = task.items[start:end]

            upsert_batch(
                batch=batch,
                data_loader=data_loader,
                items_splitter=items_splitter,
                inference_client=inference_client,
                collection=collection,
                batch_index=batch_index,
                task=task,
            )

    logger.info(f"Task {task.id} is finished.")
    task.status = UpsertionStatus.done
    context.upsertion_task.update(obj=task)
