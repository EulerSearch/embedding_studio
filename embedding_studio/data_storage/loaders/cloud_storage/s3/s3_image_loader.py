import io
from typing import Optional

from datasets import Features
from PIL import Image

from embedding_studio.data_storage.loaders.cloud_storage.s3.s3_loader import (
    AwsS3DataLoader,
)
from embedding_studio.workers.fine_tuning.utils.config import RetryConfig


class AwsS3ImageLoader(AwsS3DataLoader):
    def __init__(
        self,
        retry_config: Optional[RetryConfig] = None,
        features: Optional[Features] = None,
        **kwargs
    ):
        """Images loader from AWS S3.

        :param retry_config: retry strategy (default: None)
        :param features: expected features (default: None)
        :param kwargs: dict data for AwsS3Credentials
        """
        super(AwsS3ImageLoader, self).__init__(
            retry_config, features, **kwargs
        )

    def _get_item(self, file: io.BytesIO) -> Image:
        return Image.open(file)
