import logging
from typing import Dict, List, Set, Union

from datasets import DatasetDict

from embedding_studio.clickstream_storage.parsers import ClickstreamParser
from embedding_studio.clickstream_storage.query_retriever import QueryRetriever
from embedding_studio.clickstream_storage.raw_session import (
    RawClickstreamSession,
)
from embedding_studio.clickstream_storage.session import ClickstreamSession
from embedding_studio.data_storage.loaders.data_loader import DataLoader
from embedding_studio.data_storage.loaders.item_meta import ItemMeta
from embedding_studio.embeddings.data.clickstream.splitter import (
    ClickstreamSessionsSplitter,
)
from embedding_studio.embeddings.data.ranking_data import RankingData
from embedding_studio.embeddings.data.storages.producer import (
    ItemStorageProducer,
)
from embedding_studio.models.clickstream.sessions import SessionWithEvents

logger = logging.getLogger(__name__)


def prepare_data(
    clickstream_sessions: List[Union[Dict, SessionWithEvents]],
    parser: ClickstreamParser,
    clickstream_splitter: ClickstreamSessionsSplitter,
    query_retriever: QueryRetriever,
    loader: DataLoader,
    storage_producer: ItemStorageProducer,
) -> RankingData:
    """Prepare fine-tuning data.

    :param clickstream_sessions: clickstream sessions
    :param parser: how to parse a clickstream session
    :param clickstream_splitter: how to split clickstream sessions
    :param query_retriever: retrieve query item
    :param loader: load items data
    :param storage_producer: get train/test datasets
    :return: train / test clickstream sessiobs and dataset dict
    """
    if len(clickstream_sessions) == 0:
        raise ValueError("Empty clickstream sessions list")

    logger.info("Parse clickstream sessions data")
    raw_clickstream_sessions: List[RawClickstreamSession] = [
        (
            parser.parse(session)
            if isinstance(session, dict)
            else parser.parse_from_mongo(session)
        )
        for session in clickstream_sessions
    ]

    clickstream_sessions: List[ClickstreamSession] = [
        r.get_session() for r in raw_clickstream_sessions
    ]

    logger.info("Setup query retriever")
    query_retriever.setup(clickstream_sessions)

    logger.info("Split clickstream sessions into train / test")
    clickstream_dataset = clickstream_splitter.split(clickstream_sessions)
    logger.info(
        f'Splitting is finished, train: {len(clickstream_dataset["train"])} / test: {len(clickstream_dataset["test"])}'
    )

    logger.info("Get list of files to be loaded")
    files_to_load: Set[ItemMeta] = set()
    for session in raw_clickstream_sessions:
        files_to_load.update(set([r.item for r in session.results]))

    if len(files_to_load) == 0:
        raise ValueError("Empty clickstream sessions")

    logger.info("Download files and prepare DataDict of ItemStorage values")
    files_to_load: List[ItemMeta] = list(files_to_load)

    dataset: DatasetDict = storage_producer(
        loader.load(files_to_load), clickstream_dataset
    )

    return RankingData(clickstream_dataset, dataset)
