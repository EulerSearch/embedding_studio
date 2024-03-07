import io
from typing import Optional

from datasets import Features

from embedding_studio.data_storage.loaders.s3.s3_loader import AwsS3DataLoader
from embedding_studio.workers.fine_tuning.utils.config import RetryConfig


class AwsS3TextLoader(AwsS3DataLoader):
    def __init__(
        self,
        retry_config: Optional[RetryConfig] = None,
        features: Optional[Features] = None,
        encoding: str = "utf-8",
        **kwargs
    ):
        """Texts loader from AWS S3.

        :param retry_config: retry strategy (default: None)
        :param features: expected features (default: None)
        :param encoding: expected encoding
        :param kwargs: dict data for AwsS3Credentials
        """
        super(AwsS3TextLoader, self).__init__(retry_config, features, **kwargs)
        self.encoding = encoding

    def _get_item(self, file: io.BytesIO) -> str:
        return file.read().decode(self.encoding)
