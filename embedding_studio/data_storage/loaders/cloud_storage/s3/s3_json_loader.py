import io
import json
from typing import Dict, List, Optional, Set, Union

from datasets import Features

from embedding_studio.data_storage.loaders.cloud_storage.s3.s3_text_loader import (
    AwsS3TextLoader,
)
from embedding_studio.workers.fine_tuning.utils.config import RetryConfig


class AwsS3JSONLoader(AwsS3TextLoader):
    """
    JSON loader for AWS S3 storage.

    This class extends AwsS3TextLoader to provide specialized handling
    for JSON files stored in S3 buckets. It can also filter JSON fields
    to keep only relevant data.
    """

    def __init__(
        self,
        fields_to_keep: Optional[Union[List[str], Set[str]]] = None,
        retry_config: Optional[RetryConfig] = None,
        features: Optional[Features] = None,
        encoding: str = "utf-8",
        **kwargs
    ):
        """
        Initialize the AWS S3 JSON loader.

        :param fields_to_keep: List or set of field names to keep in the JSON (default: None)
        :param retry_config: Retry strategy configuration (default: None)
        :param features: Expected features for dataset (default: None)
        :param encoding: Character encoding for text files (default: "utf-8")
        :param kwargs: Dict data for AwsS3Credentials
        """
        super(AwsS3JSONLoader, self).__init__(
            retry_config, features, encoding, **kwargs
        )
        self.fields_to_keep = fields_to_keep

    def _filter_fields(self, item: Dict) -> Dict:
        """
        Filters an item's fields based on fields_to_keep.

        Creates a new dictionary containing only the fields that are in fields_to_keep.

        :param item: The dictionary object to filter
        :return: A new dictionary with only the specified fields
        """
        return {k: v for k, v in item.items() if k in self.fields_to_keep}

    def _get_item(self, file: io.BytesIO) -> str:
        """
        Processes a downloaded file BytesIO object into a JSON object.

        Overrides the parent's _get_item method to handle JSON processing
        and optional field filtering.

        :param file: The downloaded file as BytesIO
        :return: Decoded and parsed JSON object (dict or list)

        Example usage:
        ```python
        # Load JSON files keeping only 'id' and 'text' fields
        loader = AwsS3JSONLoader(
            fields_to_keep=["id", "text"],
            aws_access_key_id="YOUR_KEY",
            aws_secret_access_key="YOUR_SECRET"
        )
        items = loader.load_items([S3FileMeta(bucket="my-bucket", file="data.json")])
        ```
        """
        item = json.loads(super()._get_item(file))
        if self.fields_to_keep is not None:
            if isinstance(item, list):
                item = [self._filter_fields(subitem) for subitem in item]
            else:
                item = self._filter_fields(item)

        return item
