import io
from typing import Optional

from datasets import Features
from PIL import Image

from embedding_studio.data_storage.loaders.cloud_storage.gcp.gcp_loader import (
    GCPDataLoader,
)
from embedding_studio.workers.fine_tuning.utils.config import RetryConfig


class GCPImageLoader(GCPDataLoader):
    def __init__(
        self,
        retry_config: Optional[RetryConfig] = None,
        features: Optional[Features] = None,
        **kwargs
    ):
        """
        Images loader from Google Cloud Platform's Cloud Storage.

        :param retry_config: Retry strategy configuration (default: None)
        :param features: Expected schema for the images (default: None)
        :param kwargs: Keyword arguments for GcpCredentials
        """
        super(GCPImageLoader, self).__init__(retry_config, features, **kwargs)

    def _get_item(self, file: io.BytesIO) -> Image:
        """
        Extract an image from the given file object.

        :param file: io.BytesIO object containing the image data.
        :return: PIL Image object created from the data.
        """
        file.seek(0)  # Ensure the read pointer is at the start
        return Image.open(file)
