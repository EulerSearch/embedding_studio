import logging
from typing import Dict, Optional

from datasets import Features

from embedding_studio.data_storage.loaders.sql.pgsql.pgsql_loader import (
    PgsqlDataLoader,
)
from embedding_studio.workers.fine_tuning.utils.config import RetryConfig

logger = logging.getLogger(__name__)


class PgsqlTextLoader(PgsqlDataLoader):
    def __init__(
        self,
        connection_string: str,
        text_column: str = "text_data",
        retry_config: Optional[RetryConfig] = None,
        features: Optional[Features] = None,
        encoding: str = "utf-8",
        **kwargs,
    ):
        """Texts loader from PostgreSQL.

        :param connection_string: PostgreSQL connection string.
        :param text_column: The column in the database where text data is stored (default: 'text_data').
        :param retry_config: Retry strategy (default: None).
        :param features: Expected features for the dataset (default: None).
        :param encoding: The encoding to use when reading text data (default: 'utf-8').
        """
        super(PgsqlTextLoader, self).__init__(
            connection_string, retry_config, features, **kwargs
        )
        self.text_column = text_column
        self.encoding = encoding

    def _get_item(self, data: Dict) -> str:
        """Converts the text data to a string using the specified encoding.

        :param data: A dictionary containing the data for a single row, including the text data.
        :return: A string containing the decoded text data.
        """
        text_data = data.get(self.text_column)
        if text_data is None:
            logger.error(
                f"Text data not found in column '{self.text_column}'."
            )
            raise ValueError("Text data is missing.")

        # Assuming text_data is already in string format; decode if needed based on data type
        if isinstance(text_data, bytes):
            return text_data.decode(self.encoding)
        return {"item": text_data}
