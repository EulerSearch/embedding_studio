import io
from typing import Optional

from datasets import Features

from embedding_studio.data_storage.loaders.cloud_storage.gcp.gcp_loader import (
    GCPDataLoader,
)
from embedding_studio.workers.fine_tuning.utils.config import RetryConfig


class GCPTextLoader(GCPDataLoader):
    def __init__(
        self,
        retry_config: Optional[RetryConfig] = None,
        features: Optional[Features] = None,
        encoding: str = "utf-8",
        **kwargs
    ):
        """
        Text loader from Google Cloud Platform's Cloud Storage.

        :param retry_config: Retry strategy configuration (default: None).
        :param features: Schema for the data expected from the text files (default: None).
        :param encoding: Character encoding of the text files (default: "utf-8").
        :param kwargs: Additional keyword arguments for GcpCredentials.
        """
        super(GCPTextLoader, self).__init__(retry_config, features, **kwargs)
        self.encoding = encoding

    def _get_item(self, file: io.BytesIO) -> str:
        """
        Extract text from the given file object.

        :param file: io.BytesIO object containing the text data.
        :return: A string extracted from the file, decoded using the specified encoding.
        """
        file.seek(0)  # Ensure the read pointer is at the start of the file.
        return file.read().decode(self.encoding)
