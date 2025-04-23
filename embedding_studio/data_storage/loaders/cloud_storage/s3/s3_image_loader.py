import io
from typing import Optional

from datasets import Features
from PIL import Image

from embedding_studio.data_storage.loaders.cloud_storage.s3.s3_loader import (
    AwsS3DataLoader,
)
from embedding_studio.workers.fine_tuning.utils.config import RetryConfig


class AwsS3ImageLoader(AwsS3DataLoader):
    """
    Image loader for AWS S3 storage.

    This class extends AwsS3DataLoader to provide specialized handling
    for image files stored in S3 buckets, loading them as PIL Image objects.
    """

    def __init__(
        self,
        retry_config: Optional[RetryConfig] = None,
        features: Optional[Features] = None,
        **kwargs
    ):
        """
        Initialize the AWS S3 image loader.

        :param retry_config: Retry strategy configuration (default: None)
        :param features: Expected features for dataset (default: None)
        :param kwargs: Dict data for AwsS3Credentials
        """
        super(AwsS3ImageLoader, self).__init__(
            retry_config, features, **kwargs
        )

    def _get_item(self, file: io.BytesIO) -> Image:
        """
        Processes a downloaded file BytesIO object into a PIL Image.

        Overrides the parent's _get_item method to handle image loading
        using the PIL library.

        :param file: The downloaded file as BytesIO
        :return: PIL Image object

        Example usage:
        ```python
        # Load image files from S3
        loader = AwsS3ImageLoader(
            aws_access_key_id="YOUR_KEY",
            aws_secret_access_key="YOUR_SECRET"
        )

        # Get the images as PIL Image objects
        images = loader.load_items([
            S3FileMeta(bucket="my-images", file="photo1.jpg"),
            S3FileMeta(bucket="my-images", file="photo2.png")
        ])

        # Process the first image
        image = images[0].data
        resized = image.resize((300, 200))
        ```
        """
        return Image.open(file)
