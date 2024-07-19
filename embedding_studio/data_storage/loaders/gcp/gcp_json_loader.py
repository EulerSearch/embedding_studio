import io
import json
from typing import Dict, List, Optional, Set, Union

from datasets import Features

from embedding_studio.data_storage.loaders.gcp.gcp_text_loader import (
    GCPTextLoader,
)
from embedding_studio.workers.fine_tuning.utils.config import RetryConfig


class GCPJSONLoader(GCPTextLoader):
    def __init__(
        self,
        fields_to_keep: Optional[Union[List[str], Set[str]]] = None,
        retry_config: Optional[RetryConfig] = None,
        features: Optional[Features] = None,
        encoding: str = "utf-8",
        **kwargs
    ):
        """
        JSON loader from Google Cloud Platform's Cloud Storage.

        :param fields_to_keep: List or set of fields to be retained in the loaded JSON (default: None).
        :param retry_config: Retry strategy configuration (default: None).
        :param features: Schema for the data expected from the JSON (default: None).
        :param encoding: Character encoding of the JSON files (default: "utf-8").
        :param kwargs: Additional keyword arguments for GcpCredentials.
        """
        super(GCPJSONLoader, self).__init__(
            retry_config, features, encoding, **kwargs
        )
        self.fields_to_keep = fields_to_keep

    def _filter_fields(self, item: Dict) -> Dict:
        """
        Filter the JSON object to retain only the specified fields.

        :param item: The JSON object to filter.
        :return: A filtered JSON object containing only the fields specified in fields_to_keep.
        """
        if self.fields_to_keep is not None:
            return {k: v for k, v in item.items() if k in self.fields_to_keep}
        return item

    def _get_item(self, file: io.BytesIO) -> Union[Dict, List[Dict]]:
        """
        Extract and optionally filter a JSON object from the given file object.

        :param file: io.BytesIO object containing the JSON data.
        :return: A JSON object, or a list of JSON objects, potentially filtered to include only certain fields.
        """
        file.seek(0)  # Ensure the read pointer is at the start of the file.
        item = json.loads(file.read().decode(self.encoding))
        if self.fields_to_keep:
            if isinstance(item, list):
                return [self._filter_fields(subitem) for subitem in item]
            else:
                return self._filter_fields(item)
        return item
