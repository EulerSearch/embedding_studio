import io
import logging
from typing import Dict, Optional

from datasets import Features
from PIL import Image

from embedding_studio.data_storage.loaders.sql.pgsql.pgsql_loader import (
    PgsqlDataLoader,
)
from embedding_studio.workers.fine_tuning.utils.config import RetryConfig

logger = logging.getLogger(__name__)


class PgsqlImageLoader(PgsqlDataLoader):
    def __init__(
        self,
        connection_string: str,
        image_column: str = "image_data",
        retry_config: Optional[RetryConfig] = None,
        features: Optional[Features] = None,
        **kwargs,
    ):
        """Images loader from PostgreSQL.

        :param connection_string: PostgreSQL connection string.
        :param image_column: The column in the database where image data is stored (default: 'image_data').
        :param retry_config: Retry strategy (default: None).
        :param features: Expected features for the dataset (default: None).
        """
        super(PgsqlImageLoader, self).__init__(
            connection_string, retry_config, features, **kwargs
        )
        self.image_column = image_column

    def _get_item(self, data: Dict) -> Image.Image:
        """Converts the binary image data to a PIL Image object.

        :param data: A dictionary containing the data for a single row, including the image binary data.
        :return: A PIL Image object.
        """
        image_data = data.get(self.image_column)
        if image_data is None:
            logger.error(
                f"Image data not found in column '{self.image_column}'."
            )
            raise ValueError("Image data is missing.")

        return {"item": Image.open(io.BytesIO(image_data))}
