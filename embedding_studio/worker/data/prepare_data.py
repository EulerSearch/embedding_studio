import logging
from typing import Dict, List, Set

from datasets import DatasetDict

from embedding_studio.embeddings import ClickstreamSession, RankingData
from embedding_studio.embeddings.data.clickstream.parsers.parser import (
    ClickstreamParser,
)
from embedding_studio.embeddings.data.clickstream.query_retriever import (
    QueryRetriever,
)
from embedding_studio.embeddings.data.clickstream.raw_session import (
    RawClickstreamSession,
)
from embedding_studio.embeddings.data.clickstream.splitter import (
    ClickstreamSessionsSplitter,
)
from embedding_studio.embeddings.data.loaders.data_loader import DataLoader
from embedding_studio.embeddings.data.loaders.item_meta import ItemMeta
from embedding_studio.embeddings.data.storages.producer import (
    ItemStorageProducer,
)

logger = logging.getLogger(__name__)


def prepare_data(
    clickstream_sessions: List[Dict],
    parser: ClickstreamParser,
    clickstream_splitter: ClickstreamSessionsSplitter,
    query_retriever: QueryRetriever,
    loader: DataLoader,
    storage_producer: ItemStorageProducer,
) -> RankingData:
    """Prepare fine-tuning data.

    :param clickstream_sessions: clickstream sessions
    :type clickstream_sessions: List[Dict]
    :param parser: how to parse a clickstream session
    :type parser: ClickstreamParser
    :param clickstream_splitter: how to split clicstream sessions
    :type clickstream_splitter: ClickstreamSessionsSplitter
    :param query_retriever: retrieve query item
    :type query_retriever: QueryRetriever
    :param loader: load items data
    :type loader: DataLoader
    :param storage_producer: get train/test datasets
    :type storage_producer: ItemStorageProducer
    :return: train / test clickstream sessiobs and dataset dict
    :rtype: RankingData
    """
    raw_clickstream_sessions: List[RawClickstreamSession] = [
        parser.parse(session) for session in clickstream_sessions
    ]
    clickstream_sessions: List[ClickstreamSession] = [
        r.get_session() for r in raw_clickstream_sessions
    ]
    query_retriever.setup(clickstream_sessions)

    clickstream_dataset = clickstream_splitter.split(clickstream_sessions)

    files_to_load: Set[ItemMeta] = set()
    for session in raw_clickstream_sessions:
        files_to_load.update(set([r.item for r in session.results]))

    files_to_load: List[ItemMeta] = list(files_to_load)

    dataset: DatasetDict = storage_producer(
        loader.load(files_to_load), clickstream_dataset
    )
    return RankingData(clickstream_dataset, dataset)
