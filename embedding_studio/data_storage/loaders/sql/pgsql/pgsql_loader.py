import logging
from typing import Any, Dict, Generator, List, Optional, Tuple, Type

from datasets import Dataset, Features
from sqlalchemy import create_engine
from sqlalchemy.engine.row import Row
from sqlalchemy.exc import SQLAlchemyError

from embedding_studio.core.config import settings
from embedding_studio.data_storage.loaders.data_loader import DataLoader
from embedding_studio.data_storage.loaders.downloaded_item import (
    DownloadedItem,
)
from embedding_studio.data_storage.loaders.sql.pgsql.item_meta import (
    PgsqlFileMeta,
)
from embedding_studio.data_storage.loaders.sql.query_generator import (
    AbstractQueryGenerator,
)
from embedding_studio.utils.retry import retry_method
from embedding_studio.workers.fine_tuning.utils.config import (
    RetryConfig,
    RetryParams,
)

logger = logging.getLogger(__name__)


class PgsqlDataLoader(DataLoader):
    """Base loader class for retrieving data from PostgreSQL databases.

    This class provides functionality to load data from PostgreSQL databases
    using SQLAlchemy. It supports fetching individual items, batches, and
    complete datasets with configurable retry logic.

    :param connection_string: PostgreSQL connection string
    :param query_generator: PostgreSQL query generator class
    :param retry_config: Retry strategy (default: None)
    :param features: Expected features for the dataset (default: None)
    :return: A new PgsqlDataLoader instance
    """

    def __init__(
        self,
        connection_string: str,
        query_generator: Type[AbstractQueryGenerator],
        retry_config: Optional[RetryConfig] = None,
        features: Optional[Features] = None,
        **kwargs,
    ):
        """Items loader from PostgreSQL.

        :param connection_string: PostgreSQL connection string.
        :param query_generator: PostgreSQL query generator class.
        :param retry_config: retry strategy (default: None).
        :param features: expected features (default: None).
        """
        super(PgsqlDataLoader, self).__init__(**kwargs)
        self.connection_string = connection_string
        self.retry_config = (
            retry_config if retry_config else self._get_default_retry_config()
        )
        self.features = features
        self.engine = create_engine(
            self.connection_string,
        )
        self.query_generator = query_generator(self.engine)

    @property
    def item_meta_cls(self) -> Type[PgsqlFileMeta]:
        """Return the item metadata class used by this loader.

        :return: The PgsqlFileMeta class
        """
        return PgsqlFileMeta

    @staticmethod
    def _get_default_retry_config() -> RetryConfig:
        """Create default retry configuration for PostgreSQL operations.

        :return: A RetryConfig object with default settings
        """
        default_retry_params = RetryParams(
            max_attempts=settings.DEFAULT_MAX_ATTEMPTS,
            wait_time_seconds=settings.DEFAULT_WAIT_TIME_SECONDS,
        )

        config = RetryConfig(default_params=default_retry_params)
        config["fetch_data"] = RetryParams(
            max_attempts=settings.PGSQL_DATA_LOADER_ATTEMPTS,
            wait_time_seconds=settings.PGSQL_DATA_LOADER_WAIT_TIME_SECONDS,
        )
        return config

    @retry_method(name="fetch_data")
    def _fetch_data(self, row_ids: List[int]) -> List[Dict[str, Any]]:
        """Fetch data from PostgreSQL based on a list of row IDs.

        :param row_ids: The list of row IDs to fetch data for.
        :return: A list of dictionaries containing the data for the given row IDs.
        """
        try:
            with self.engine.connect() as connection:
                query = self.query_generator.fetch_all(
                    row_ids
                )  # Using fetch_all with multiple row_ids
                results = connection.execute(query).fetchall()

                return (
                    [
                        dict(row._mapping)
                        if isinstance(row, Row)
                        else dict(row)
                        for row in results
                    ]
                    if results
                    else []
                )
        except SQLAlchemyError as e:
            logger.exception(
                f"Failed to fetch data for row IDs {row_ids}: {str(e)}"
            )
            return []

    def _get_item(self, data: Any) -> Any:
        """Process the raw data from PostgreSQL into the required format.

        This method should be overridden by subclasses to transform the raw data
        into the desired format.

        Example implementation:
        ```python
        def _get_item(self, data: Any) -> Dict:
            return {
                "processed_item": self._process_data(data),
                "timestamp": data.get("created_at")
            }
        ```

        :param data: The raw data fetched from PostgreSQL
        :return: Processed data in the required format
        """
        return data

    def _get_data_from_db(
        self,
        items_data: List[PgsqlFileMeta],
        ignore_failures: bool = True,
    ) -> Generator[Tuple[Dict, PgsqlFileMeta], None, None]:
        """Main method to retrieve data from PostgreSQL using a list of item metadata objects.

        :param items_data: A list of PgsqlFileMeta objects containing metadata about each row to fetch.
        :param ignore_failures: If True, continues with the next items after a failure; otherwise, raises an exception.
        :return: A generator of dictionaries, each containing the data and its corresponding metadata.
        """
        row_ids = [item_meta.object_id for item_meta in items_data]

        try:
            data_list = self._fetch_data(row_ids)  # Fetch all data at once

            # Create a mapping from row ID to data for fast lookup
            data_map = {data["id"]: data for data in data_list}

            for item_meta in items_data:
                data = data_map.get(item_meta.object_id)
                if data:
                    yield self._create_item_object(data, item_meta)
                else:
                    logger.error(
                        f"No data found for row ID {item_meta.object_id}"
                    )
        except Exception as e:
            logger.exception(f"Failed to fetch batch data: {str(e)}")
            if not ignore_failures:
                raise

    def _create_item_object(
        self, data: Dict, item_meta: PgsqlFileMeta
    ) -> Tuple[Dict, PgsqlFileMeta]:
        """Creates a dictionary object from the fetched data and includes metadata.

        :param data: The data content of the item.
        :param item_meta: The metadata of the item.
        :return:  A tuple containing the item dictionary and its metadata.
        """
        return (
            {
                "item_id": item_meta.id,
                **self._get_item(
                    data
                ),  # Assuming _get_item returns a dictionary
            },
            item_meta,
        )

    def load(self, items_data: List[PgsqlFileMeta]) -> Dataset:
        """Loads a dataset of data from PostgreSQL.

        :param items_data: List of item metadata to load.
        :return: A Dataset object containing the data.
        """

        def data_generator():
            for item, _ in self._get_data_from_db(items_data):
                yield item  # Ensure this yields a dictionary, not a tuple

        return Dataset.from_generator(
            data_generator,
            features=self.features,
        )

    def load_items(
        self, items_data: List[PgsqlFileMeta]
    ) -> List[DownloadedItem]:
        """Loads items from PostgreSQL.

        :param items_data: List of item metadata to load.
        :return: A list of DownloadedItem objects.
        """
        result = []
        for item_object, item_meta in self._get_data_from_db(
            items_data, ignore_failures=False
        ):
            result.append(
                DownloadedItem(
                    id=item_object["item_id"],
                    data=item_object["item"]
                    if "item" in item_object
                    else item_object,
                    meta=item_meta,
                )
            )
        return result

    def _load_batch_with_offset(
        self, offset: int, batch_size: int, **kwargs
    ) -> List[DownloadedItem]:
        """Load a batch of rows from PostgreSQL starting from the given offset up to the batch size.

        :param offset: The offset from where to start loading rows.
        :param batch_size: The number of rows to load.
        :return: A list of downloaded items, each containing the row ID, its content, and metadata.
        """
        logger.info(
            f"Loading batch from PostgreSQL with offset {offset} and batch size {batch_size}"
        )
        try:
            with self.engine.connect() as connection:
                query = self.query_generator.all(
                    offset=offset, batch_size=batch_size
                )
                results = connection.execute(query).fetchall()

                batch = []
                for result in results:
                    data = dict(result)
                    item_meta = PgsqlFileMeta(object_id=data.get("id"))
                    batch.append(
                        DownloadedItem(
                            id=item_meta.id, data=data, meta=item_meta
                        )
                    )
                return batch
        except SQLAlchemyError as e:
            logger.exception(f"Error fetching batch: {str(e)}")
            return []

    def load_all(
        self, batch_size: int, **kwargs
    ) -> Generator[DownloadedItem, None, None]:
        """A generator that iteratively loads batches from PostgreSQL using the `load_batch` method.

        :param batch_size: The size of each batch to load.
        :yield: Each batch as a list of downloaded items (id, data, item_info).
        """
        offset = 0
        while True:
            current_batch = self._load_batch_with_offset(offset, batch_size)
            if not current_batch:
                break  # Stop yielding if no more data is returned.
            yield current_batch
            offset += batch_size

    def total_count(self) -> Optional[int]:
        """Returns the total count of rows in the specified table.

        :return: The total number of rows or None if the count fails.
        """
        try:
            with self.engine.connect() as connection:
                query = self.query_generator.count()
                result = connection.execute(query).scalar()
                return result
        except SQLAlchemyError as e:
            logger.exception(f"Error fetching total count: {str(e)}")
            return None
