import io
from typing import Optional

from datasets import Features

from embedding_studio.data_storage.loaders.cloud_storage.s3.s3_loader import (
    AwsS3DataLoader,
)
from embedding_studio.workers.fine_tuning.utils.config import RetryConfig


class AwsS3TextLoader(AwsS3DataLoader):
    """
    Text loader for AWS S3 storage.

    This class extends AwsS3DataLoader to provide specialized handling
    for text files stored in S3 buckets with configurable encoding.
    """

    def __init__(
        self,
        retry_config: Optional[RetryConfig] = None,
        features: Optional[Features] = None,
        encoding: str = "utf-8",
        **kwargs
    ):
        """
        Initialize the AWS S3 text loader.

        :param retry_config: Retry strategy configuration (default: None)
        :param features: Expected features for dataset (default: None)
        :param encoding: Character encoding for text files (default: "utf-8")
        :param kwargs: Dict data for AwsS3Credentials
        """
        super(AwsS3TextLoader, self).__init__(retry_config, features, **kwargs)
        self.encoding = encoding

    def _get_item(self, file: io.BytesIO) -> str:
        """
        Processes a downloaded file BytesIO object into a string.

        Overrides the parent's _get_item method to handle text decoding
        with the specified encoding.

        :param file: The downloaded file as BytesIO
        :return: Decoded text content as a string

        Example usage:
        ```python
        # Load UTF-8 text files from S3
        loader = AwsS3TextLoader(
            encoding="utf-8",
            aws_access_key_id="YOUR_KEY",
            aws_secret_access_key="YOUR_SECRET"
        )
        items = loader.load_items([S3FileMeta(bucket="my-bucket", file="document.txt")])
        ```
        """
        return file.read().decode(self.encoding)
