import logging
from typing import Dict, List, Set, Union

from embedding_studio.clickstream_storage.converters.converter import (
    ClickstreamSessionConverter,
)
from embedding_studio.clickstream_storage.input_with_items import (
    FineTuningInputWithItems,
)
from embedding_studio.clickstream_storage.query_retriever import QueryRetriever
from embedding_studio.data_storage.loaders.downloaded_item import (
    DownloadedItem,
)
from embedding_studio.data_storage.loaders.data_loader import DataLoader
from embedding_studio.data_storage.loaders.item_meta import ItemMeta
from embedding_studio.embeddings.data.clickstream.train_test_splitter import (
    TrainTestSplitter,
)
from embedding_studio.embeddings.data.ranking_data import RankingData
from embedding_studio.embeddings.data.storages.producer import (
    ItemStorageProducer,
)
from embedding_studio.embeddings.features.feature_extractor_input import (
    FineTuningInput,
)
from embedding_studio.models.clickstream.sessions import SessionWithEvents

logger = logging.getLogger(__name__)


def prepare_data(
    fine_tuning_inputs: List[Union[Dict, SessionWithEvents]],
    converter: ClickstreamSessionConverter,
    clickstream_splitter: TrainTestSplitter,
    query_retriever: QueryRetriever,
    loader: DataLoader,
    storage_producer: ItemStorageProducer,
) -> RankingData:
    """Prepare fine-tuning data.

    :param fine_tuning_inputs: clickstream inputs
    :param converter: how to converter a clickstream session into fine-tuning input
    :param clickstream_splitter: how to split clickstream inputs
    :param query_retriever: retrieve query item
    :param loader: load items data
    :param storage_producer: get train/test datasets
    :return: train / test clickstream sessiobs and dataset dict
    """
    if len(fine_tuning_inputs) == 0:
        raise ValueError("Empty clickstream inputs list")

    logger.info("Parse clickstream inputs data")
    input_with_items: List[FineTuningInputWithItems] = [
        converter.convert(session) for session in fine_tuning_inputs
    ]


    logger.info("Get list of files to be loaded")
    files_to_load: Set[ItemMeta] = set()
    for obj in input_with_items:
        files_to_load.update(set(obj.items))

    if len(files_to_load) == 0:
        raise ValueError("Empty clickstream inputs")

    logger.info("Download files and prepare DataDict of ItemStorage values")
    files_to_load: List[ItemMeta] = list(files_to_load)
    downloaded: List[DownloadedItem] = loader.load(files_to_load)
    if len(downloaded) == 0:
        raise ValueError('No data was downloaded.')

    inputs = [obj.input for obj in input_with_items]
    if len(downloaded) != len(files_to_load):
        logger.info("Remove items failed to be downloaded.")
        ids_to_load = set()
        for item in files_to_load:
            ids_to_load.add(item.id)

        downloaded_ids = set()
        for item in downloaded:
            downloaded_ids.add(item.id)

        # TODO: pass failed_ids and related exceptions to the worker status
        failed_ids = ids_to_load.difference(downloaded_ids)

        filtered_inputs = []
        for fine_tuning_input in inputs:
            fine_tuning_input.remove_results(failed_ids)

            if len(fine_tuning_input.results) >= 2:
                filtered_inputs.append(fine_tuning_input)

        inputs = filtered_inputs

    logger.info("Retrieve queries")
    query_retriever.get_queries(inputs)

    logger.info("Split clickstream inputs into train / test")
    training_dataset = clickstream_splitter.split(inputs)
    logger.info(
        f'Splitting is finished, train: {len(training_dataset["train"])} / test: {len(training_dataset["test"])}'
    )


    dataset, clickstream_dataset = storage_producer(
        downloaded, training_dataset
    )

    return RankingData(clickstream_dataset, dataset)
