import logging
from typing import Dict, List, Optional, Set, Type, Union

from datasets import Features

from embedding_studio.data_storage.loaders.sql.pgsql.pgsql_loader import (
    PgsqlDataLoader,
)
from embedding_studio.data_storage.loaders.sql.query_generator import (
    AbstractQueryGenerator,
)
from embedding_studio.workers.fine_tuning.utils.config import RetryConfig

logger = logging.getLogger(__name__)


class PgsqlJSONBLoader(PgsqlDataLoader):
    def __init__(
        self,
        connection_string: str,
        query_generator: Type[AbstractQueryGenerator],
        jsonb_column: str = "jsonb_data",
        fields_to_keep: Optional[Union[List[str], Set[str]]] = None,
        retry_config: Optional[RetryConfig] = None,
        features: Optional[Features] = None,
        **kwargs,
    ):
        """JSONB loader from PostgreSQL.

        :param connection_string: PostgreSQL connection string.
        :param query_generator: PostgreSQL query generator class.
        :param jsonb_column: The column in the database where JSONB data is stored (default: 'jsonb_data').
        :param fields_to_keep: List or set of fields to select from the JSONB data (default: None).
        :param retry_config: Retry strategy (default: None).
        :param features: Expected features for the dataset (default: None).
        """
        super(PgsqlJSONBLoader, self).__init__(
            connection_string,
            query_generator,
            retry_config,
            features,
            **kwargs,
        )
        self.jsonb_column = jsonb_column
        self.fields_to_keep = fields_to_keep

    def _get_item(self, data: Dict) -> Dict:
        """Extracts and optionally filters the JSONB data.

        :param data: A dictionary containing the data for a single row, including JSONB data.
        :return: A dictionary containing the parsed JSONB data.
        """
        jsonb_data = data.get(self.jsonb_column)
        if jsonb_data is None:
            logger.error(
                f"JSONB data not found in column '{self.jsonb_column}'."
            )
            raise ValueError("JSONB data is missing.")

        # Filter the JSON fields if fields_to_keep is specified
        if self.fields_to_keep is not None:
            if isinstance(jsonb_data, dict):
                return {
                    k: v
                    for k, v in jsonb_data.items()
                    if k in self.fields_to_keep
                }
            else:
                logger.error(
                    f"Expected dict type for JSONB data, got {type(jsonb_data)}."
                )
                raise ValueError("JSONB data format is incorrect.")

        return jsonb_data
