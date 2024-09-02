import logging
from typing import Any, Dict, Generator, List, Optional, Tuple, Type

from datasets import Dataset, Features
from sqlalchemy import create_engine, text, select, Table, MetaData
from sqlalchemy.engine.row import Row
from sqlalchemy.sql import func
from sqlalchemy.exc import SQLAlchemyError

from embedding_studio.core.config import settings
from embedding_studio.data_storage.loaders.data_loader import DataLoader
from embedding_studio.data_storage.loaders.downloaded_item import (
    DownloadedItem,
)
from embedding_studio.data_storage.loaders.sql.sql_item_meta import SQLFileMeta
from embedding_studio.workers.fine_tuning.utils.config import (
    RetryConfig,
    RetryParams,
)
from embedding_studio.workers.fine_tuning.utils.retry import retry_method

logger = logging.getLogger(__name__)


class PgsqlDataLoader(DataLoader):
    def __init__(
        self,
        connection_string: str,
        retry_config: Optional[RetryConfig] = None,
        features: Optional[Features] = None,
        **kwargs,
    ):
        """Items loader from PostgreSQL.

        :param connection_string: PostgreSQL connection string.
        :param retry_config: retry strategy (default: None).
        :param features: expected features (default: None).
        """
        super(PgsqlDataLoader, self).__init__(**kwargs)
        self.connection_string = connection_string
        self.retry_config = (
            retry_config if retry_config else self._get_default_retry_config()
        )
        self.features = features
        self.engine = create_engine(self.connection_string)

    @property
    def item_meta_cls(self) -> Type[SQLFileMeta]:
        return SQLFileMeta

    @staticmethod
    def _get_default_retry_config() -> RetryConfig:
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
    def _fetch_data(self, row_id: int, table_name: str) -> Any:
        """Fetch data from PostgreSQL based on row ID.

        :param row_id: The row ID to fetch data for.
        :param table_name: The name of the table to query.
        :return: The data for the given row ID as a dictionary.
        """
        try:
            with self.engine.connect() as connection:
                metadata = MetaData(bind=self.engine)
                table = Table(table_name, metadata, autoload_with=self.engine)
                query = select(table).where(table.c.id == row_id)
                result = connection.execute(query).fetchone()

                if isinstance(result, Row):
                    return dict(result._mapping)
                elif result:
                    return {key: value for key, value in result.items()}
                return None
        except SQLAlchemyError as e:
            logger.exception(
                f"Failed to fetch data from {table_name} for row ID {row_id}: {str(e)}"
            )
            return None

    def _get_item(self, data: Any) -> Any:
        return data

    def _get_data_from_db(
        self,
        items_data: List[SQLFileMeta],
        table_name: str,
        ignore_failures: bool = True,
    ) -> Generator[Tuple[Dict, SQLFileMeta], None, None]:
        """Main method to retrieve data from PostgreSQL using a list of item metadata objects.

        :param items_data: A list of SQLFileMeta objects containing metadata about each row to fetch.
        :param table_name: The name of the table to query.
        :param ignore_failures: If True, continues with the next items after a failure; otherwise, raises an exception.
        :return: A generator of dictionaries, each containing the data and its corresponding metadata.
        """
        for item_meta in items_data:
            try:
                data = self._fetch_data(item_meta.object_id, table_name)
                if data is not None:
                    # Yield the dictionary directly, not as part of a tuple
                    yield self._create_item_object(data, item_meta)
                else:
                    logger.error(
                        f"No data found for row ID {item_meta.object_id} in table {table_name}"
                    )
            except Exception as e:
                logger.exception(
                    f"Failed to process row ID {item_meta.object_id} in table {table_name}: {str(e)}"
                )
                if not ignore_failures:
                    raise

    def _create_item_object(
        self, data: Dict, item_meta: SQLFileMeta
    ) -> Tuple[Dict, SQLFileMeta]:
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

    def load(self, items_data: List[SQLFileMeta], table_name: str) -> Dataset:
        """Loads a dataset of data from PostgreSQL.

        :param items_data: List of item metadata to load.
        :param table_name: The name of the table to load data from.
        :return: A Dataset object containing the data.
        """

        def data_generator():
            for item, _ in self._get_data_from_db(items_data, table_name):
                yield item  # Ensure this yields a dictionary, not a tuple

        return Dataset.from_generator(
            data_generator,
            features=self.features,
        )

    def load_items(
        self, items_data: List[SQLFileMeta], table_name: str
    ) -> List[DownloadedItem]:
        """Loads items from PostgreSQL.

        :param items_data: List of item metadata to load.
        :param table_name: The name of the table to load data from.
        :return: A list of DownloadedItem objects.
        """
        result = []
        for item_object, item_meta in self._get_data_from_db(
            items_data, table_name, ignore_failures=False
        ):
            result.append(
                DownloadedItem(
                    id=item_object["item_id"],
                    data=item_object["item"],
                    meta=item_meta,
                )
            )
        return result

    def _load_batch_with_offset(
        self, offset: int, batch_size: int, table_name: str, **kwargs
    ) -> List[DownloadedItem]:
        """Load a batch of rows from PostgreSQL starting from the given offset up to the batch size.

        :param offset: The offset from where to start loading rows.
        :param batch_size: The number of rows to load.
        :param table_name: The name of the table to load data from.
        :return: A list of downloaded items, each containing the row ID, its content, and metadata.
        """
        logger.info(
            f"Loading batch from PostgreSQL table {table_name} with offset {offset} and batch size {batch_size}"
        )
        try:
            with self.engine.connect() as connection:
                metadata = MetaData(bind=self.engine)
                table = Table(table_name, metadata, autoload_with=self.engine)
                query = select(table).order_by(table.c.id).limit(batch_size).offset(offset)
                results = connection.execute(query).fetchall()

                batch = []
                for result in results:
                    data = dict(result)
                    item_meta = SQLFileMeta(object_id=data.get("id"))
                    batch.append(
                        DownloadedItem(
                            id=item_meta.id, data=data, meta=item_meta
                        )
                    )
                return batch
        except SQLAlchemyError as e:
            logger.exception(
                f"Error fetching batch from table {table_name}: {str(e)}"
            )
            return []

    def load_all(
        self, batch_size: int, table_name: str, **kwargs
    ) -> Generator[DownloadedItem, None, None]:
        """A generator that iteratively loads batches from PostgreSQL using the `load_batch` method.

        :param batch_size: The size of each batch to load.
        :param table_name: The name of the table to load data from.
        :yield: Each batch as a list of downloaded items (id, data, item_info).
        """
        offset = 0
        while True:
            current_batch = self._load_batch_with_offset(
                offset, batch_size, table_name
            )
            if not current_batch:
                break  # Stop yielding if no more data is returned.
            yield current_batch
            offset += batch_size

    def total_count(self, table_name: str) -> Optional[int]:
        """Returns the total count of rows in the specified table.

        :param table_name: The name of the table to count rows in.
        :return: The total number of rows or None if the count fails.
        """
        try:
            with self.engine.connect() as connection:
                metadata = MetaData(bind=self.engine)
                table = Table(table_name, metadata, autoload_with=self.engine)
                query = select([func.count()]).select_from(table)
                result = connection.execute(query).scalar()
                return result
        except SQLAlchemyError as e:
            logger.exception(
                f"Error fetching total count from table {table_name}: {str(e)}"
            )
            return None
