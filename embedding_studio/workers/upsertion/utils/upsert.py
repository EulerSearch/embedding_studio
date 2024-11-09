import logging
import traceback
from typing import List, Tuple

from embedding_studio.context.app_context import context
from embedding_studio.core.config import settings
from embedding_studio.core.plugin import PluginManager
from embedding_studio.data_access.mongo.crud_base import CRUDBase
from embedding_studio.data_storage.loaders.data_loader import DataLoader
from embedding_studio.embeddings.inference.triton.client import TritonClient
from embedding_studio.embeddings.splitters.item_splitter import ItemSplitter
from embedding_studio.models.items_handler import (
    BaseDataHandlingTask,
    DataItem,
    ItemProcessingFailureStage,
)
from embedding_studio.models.task import TaskStatus
from embedding_studio.models.utils import create_failed_data_item
from embedding_studio.vectordb.collection import Collection
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


def handle_failed_items(
    failed_items: List[Tuple[DataItem, str]],
    task: BaseDataHandlingTask,
    exception: Exception,
    task_crud: CRUDBase,
):
    """
    Handles failed items during different stages of the upsertion process.

    :param failed_items: List of tuples containing failed DataItem and traceback.
    :param task: The upsertion task object in the database.
    :param exception: The exception that occurred.
    :param task_crud: The CRUD object that contains information about failed items.
    """
    stage = None
    stage_description = ""
    if isinstance(exception, DownloadException):
        stage = ItemProcessingFailureStage.on_downloading
        stage_description = "downloading items batch"

    elif isinstance(exception, SplitException):
        stage = ItemProcessingFailureStage.on_splitting
        stage_description = "splitting"

    elif isinstance(exception, InferenceException):
        stage = ItemProcessingFailureStage.on_inference
        stage_description = "running inference"

    elif isinstance(exception, UploadException):
        stage = ItemProcessingFailureStage.on_upsert
        stage_description = "uploading vectors in DB"

    message = (
        f"Something went wrong during {stage_description}"
        f" for {len(failed_items)} items [task ID: {task.id}]"
    )
    logger.exception(message)

    for item, tb in failed_items:
        task.failed_items.append(create_failed_data_item(item, tb, stage))

    context.upsertion_task.update(obj=task)
    if not settings.UPSERTION_IGNORE_FAILED_ITEMS:
        task.status = TaskStatus.failed
        task_crud.update(obj=task)
        raise ValueError(message)


def upsert_batch(
    batch: List[DataItem],
    data_loader: DataLoader,
    items_splitter: ItemSplitter,
    inference_client: TritonClient,
    collection: Collection,
    batch_index: int,
    task: BaseDataHandlingTask,
    task_crud: CRUDBase,
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
    :param task_crud: The CRUD object that contains information about task.
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
            f"Split items data for {batch_index} batch "
            f"with {len(downloaded_items)} items in it [task ID: {task.id}]"
        )
        parts, object_to_parts, failed = split_items(
            downloaded_items, items_splitter
        )
        logger.info(
            f"Split result for {batch_index} batch: {len(downloaded_items)} "
            f"items -> {len(parts)} parts, [task ID: {task.id}]"
        )

        if len(failed) > 0:
            handle_failed_items(
                failed_items=[
                    (batch[id_to_index[item.meta.object_id]], tb)
                    for item, tb in failed
                ],
                task=task,
                exception=SplitException(),
                task_crud=task_crud,
            )

        logger.info(
            f"Run inference for {batch_index} batch with {len(parts)} "
            f"parts in total [task ID: {task.id}]"
        )
        vectors = run_inference(parts, inference_client)

        logger.info(
            f"Upload vectors for {batch_index} batch "
            f"[dims: {vectors.shape}] [task ID: {task.id}]"
        )
        upload_vectors(
            items=downloaded_items,
            vectors=vectors,
            object_to_parts=object_to_parts,
            collection=collection,
        )

    except Exception as e:
        tb = traceback.format_exc()[-1500:]
        handle_failed_items(
            failed_items=[(item, tb) for item in batch],
            task=task,
            exception=e,
            task_crud=task_crud,
        )


def process_upsert(
    task: BaseDataHandlingTask,
    collection: Collection,
    data_loader: DataLoader,
    items_splitter: ItemSplitter,
    inference_client: TritonClient,
    task_crud: CRUDBase,
):
    # Extract all object IDs from the task items
    all_object_ids = [item.object_id for item in task.items]
    batches = len(task.items) // settings.UPSERTION_BATCH_SIZE + 1
    with collection.lock_objects(
        all_object_ids, max_attempts=5, wait_time=2.0
    ):
        for batch_index in range(batches):
            start = batch_index * settings.UPSERTION_BATCH_SIZE
            end = min(
                (batch_index + 1) * settings.UPSERTION_BATCH_SIZE,
                len(task.items),
            )

            if end <= len(task.items):
                batch = task.items[start:end]
                if len(batch) == 0:
                    continue

                upsert_batch(
                    batch=batch,
                    data_loader=data_loader,
                    items_splitter=items_splitter,
                    inference_client=inference_client,
                    collection=collection,
                    batch_index=batch_index,
                    task=task,
                    task_crud=task_crud,
                )

    task.status = TaskStatus.done
    task_crud.update(obj=task)
