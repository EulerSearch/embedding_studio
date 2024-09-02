import logging
from typing import Dict, List, Optional

from datasets import Features

from embedding_studio.data_storage.loaders.sql.pgsql.pgsql_loader import (
    PgsqlDataLoader,
)
from embedding_studio.workers.fine_tuning.utils.config import RetryConfig

logger = logging.getLogger(__name__)


class PgsqlMultiTextColumnLoader(PgsqlDataLoader):
    def __init__(
        self,
        connection_string: str,
        text_columns: List[str],
        retry_config: Optional[RetryConfig] = None,
        features: Optional[Features] = None,
        encoding: str = "utf-8",
        **kwargs,
    ):
        """Multi-text columns loader from PostgreSQL.

        :param connection_string: PostgreSQL connection string.
        :param text_columns: A list of columns in the database where text data is stored.
        :param retry_config: Retry strategy (default: None).
        :param features: Expected features for the dataset (default: None).
        :param encoding: The encoding to use when reading text data (default: 'utf-8').
        """
        super(PgsqlMultiTextColumnLoader, self).__init__(
            connection_string, retry_config, features, **kwargs
        )
        self.text_columns = text_columns
        self.encoding = encoding

    def _get_item(self, data: Dict) -> Dict[str, str]:
        """Extracts and decodes the text data from multiple columns.

        :param data: A dictionary containing the data for a single row, including multiple text columns.
        :return: A dictionary with the text data from the specified columns.
        """
        text_data = {}
        for column in self.text_columns:
            column_data = data.get(column)
            if column_data is None:
                logger.error(f"Text data not found in column '{column}'.")
                raise ValueError(f"Text data is missing in column '{column}'.")

            # Assuming text_data is already in string format; decode if needed based on data type
            if isinstance(column_data, bytes):
                text_data[column] = column_data.decode(self.encoding)
            else:
                text_data[column] = column_data

        return text_data
