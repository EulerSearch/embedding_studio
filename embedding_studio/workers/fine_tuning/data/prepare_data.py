import logging
from typing import Dict, List, Set, Union

from embedding_studio.clickstream_storage.parsers.parser import (
    ClickstreamParser,
)
from embedding_studio.clickstream_storage.query_retriever import QueryRetriever
from embedding_studio.clickstream_storage.raw_session import (
    RawClickstreamSession,
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
    clickstream_sessions: List[Union[Dict, SessionWithEvents]],
    parser: ClickstreamParser,
    clickstream_splitter: TrainTestSplitter,
    query_retriever: QueryRetriever,
    loader: DataLoader,
    storage_producer: ItemStorageProducer,
) -> RankingData:
    """Prepare fine-tuning data.

    :param clickstream_sessions: clickstream inputs
    :param parser: how to parse a clickstream session
    :param clickstream_splitter: how to split clickstream inputs
    :param query_retriever: retrieve query item
    :param loader: load items data
    :param storage_producer: get train/test datasets
    :return: train / test clickstream sessiobs and dataset dict
    """
    if len(clickstream_sessions) == 0:
        raise ValueError("Empty clickstream inputs list")

    logger.info("Parse clickstream inputs data")
    raw_clickstream_sessions: List[RawClickstreamSession] = [
        (
            parser.parse(session)
            if isinstance(session, dict)
            else parser.parse_from_mongo(session)
        )
        for session in clickstream_sessions
    ]

    clickstream_sessions: List[FineTuningInput] = [
        r.get_fine_tuning_input() for r in raw_clickstream_sessions
    ]

    logger.info("Retrieve queries")
    query_retriever.get_queries(clickstream_sessions)

    logger.info("Split clickstream inputs into train / test")
    clickstream_dataset = clickstream_splitter.split(clickstream_sessions)
    logger.info(
        f'Splitting is finished, train: {len(clickstream_dataset["train"])} / test: {len(clickstream_dataset["test"])}'
    )

    logger.info("Get list of files to be loaded")
    files_to_load: Set[ItemMeta] = set()
    for session in raw_clickstream_sessions:
        files_to_load.update(set([r.item for r in session.results]))

    if len(files_to_load) == 0:
        raise ValueError("Empty clickstream inputs")

    logger.info("Download files and prepare DataDict of ItemStorage values")
    files_to_load: List[ItemMeta] = list(files_to_load)

    dataset, clickstream_dataset = storage_producer(
        loader.load(files_to_load), clickstream_dataset
    )

    return RankingData(clickstream_dataset, dataset)
