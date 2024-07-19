import logging
import traceback
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np

from embedding_studio.data_storage.loaders.data_loader import DataLoader
from embedding_studio.data_storage.loaders.downloaded_item import (
    DownloadedItem,
)
from embedding_studio.embeddings.inference.triton.client import TritonClient
from embedding_studio.embeddings.splitters.item_splitter import ItemSplitter
from embedding_studio.models.embeddings.objects import Object, ObjectPart
from embedding_studio.models.upsert import DataItem
from embedding_studio.vectordb.collection import Collection
from embedding_studio.workers.upsertion.utils.exceptions import (
    DownloadException,
    InferenceException,
    SplitException,
    UploadException,
)

logger = logging.getLogger(__name__)


def download_items(
    items: List[DataItem], data_loader: DataLoader
) -> List[DownloadedItem]:
    """
    Download a list of items using the specified DataLoader.

    :param items: List of DataItem instances to download.
    :param data_loader: DataLoader instance used for downloading items.
    :return: List of DownloadedItem instances.
    """
    try:
        items_to_download = []
        for item in items:
            item_to_download = data_loader.item_meta_cls(**item.item_info)
            item_to_download.object_id = item.object_id
            item_to_download.payload = item.payload
            items_to_download.append(item_to_download)

        return data_loader.load_items(items_to_download)

    except Exception:
        raise DownloadException()


def split_items(
    items: List[DownloadedItem], item_splitter: ItemSplitter
) -> Tuple[List[Any], Dict[str, List[int]], List[Tuple[DownloadedItem, str]]]:
    """
    Split each item into parts using the specified ItemSplitter.

    :param items: List of DownloadedItem instances to split.
    :param item_splitter: ItemSplitter instance used for splitting the items.
    :return: A tuple containing a list of parts, a dictionary mapping objects to their parts,
             and a list of tuples with failed items and their traceback.
    """
    try:
        object_to_parts = defaultdict(list)
        parts = []
        failed = []
        for item in items:
            try:
                split_data = item_splitter(item.data)
                object_to_parts[item.meta.object_id] = [
                    i + len(parts) for i in range(len(split_data))
                ]
                parts += split_data
            except Exception:
                tb = traceback.format_exc()
                failed.append((item, tb))

        return parts, object_to_parts, failed

    except Exception:
        raise SplitException()


def run_inference(
    items_data: List[Any],
    inference_client: TritonClient,
) -> np.ndarray:
    """
    Run inference on the given items data using the specified TritonClient.

    :param items_data: List of data on which to run inference.
    :param inference_client: TritonClient to handle the inference process.
    :return: Array of vectors representing the inference results.
    """
    try:
        inference_batch_size = 16
        batches = len(items_data) // inference_batch_size + 1
        result = []
        for batch_index in range(batches):
            start = batch_index * inference_batch_size
            end = min(
                (batch_index + 1) * inference_batch_size,
                len(items_data),
            )
            if start < len(items_data) and start != end:
                batch_result = inference_client.forward_items(
                    items_data[start:end]
                )
                result.append(batch_result)

        return np.vstack(result)

    except Exception:
        raise InferenceException()


def upload_vectors(
    items: List[DownloadedItem],
    vectors: np.ndarray,
    object_to_parts: Dict[str, List[int]],
    collection: Collection,
):
    """
    Upload vectors to the specified collection.

    :param items: List of DownloadedItem instances whose vectors need to be uploaded.
    :param vectors: Numpy array containing the vectors to upload.
    :param object_to_parts: Dictionary mapping object IDs to the indices of their parts in the vectors array.
    :param collection: Collection instance to which vectors will be uploaded.
    """
    try:
        objects = []
        for item in items:
            parts = []
            for part_index in object_to_parts[item.meta.object_id]:
                parts.append(
                    ObjectPart(
                        vector=vectors[part_index].tolist(),
                        part_id=f"{item.meta.object_id}:{part_index}",
                    )
                )

            objects.append(
                Object(
                    object_id=item.meta.object_id,
                    parts=parts,
                    payload=item.meta.payload,
                )
            )

        collection.upsert(objects)

    except Exception:
        raise UploadException()
