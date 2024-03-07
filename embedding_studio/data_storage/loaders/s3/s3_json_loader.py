import io
import json
from typing import Dict, List, Optional, Set, Union

from datasets import Features

from embedding_studio.data_storage.loaders.s3.s3_text_loader import (
    AwsS3TextLoader,
)
from embedding_studio.workers.fine_tuning.utils.config import RetryConfig


class AwsS3JSONLoader(AwsS3TextLoader):
    def __init__(
        self,
        fields_to_keep: Optional[Union[List[str], Set[str]]] = None,
        retry_config: Optional[RetryConfig] = None,
        features: Optional[Features] = None,
        encoding: str = "utf-8",
        **kwargs
    ):
        """JSON loader from AWS S3.

        :apram fields_to_keep: list of fields to be selected (default: None)
        :param retry_config: retry strategy (default: None)
        :param features: expected features (default: None)
        :param encoding: expected encoding
        :param kwargs: dict data for AwsS3Credentials
        """
        super(AwsS3JSONLoader, self).__init__(
            retry_config, features, encoding, **kwargs
        )
        self.fields_to_keep = fields_to_keep

    def _filter_fields(self, item: Dict) -> Dict:
        return {k: v for k, v in item.items() if k in self.fields_to_keep}

    def _get_item(self, file: io.BytesIO) -> str:
        item = json.loads(super()._get_item(file))
        if self.fields_to_keep is not None:
            if isinstance(item, list):
                item = [self._filter_fields(subitem) for subitem in item]
            else:
                item = self._filter_fields(item)

        return item
