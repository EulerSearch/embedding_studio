from typing import Any

from pydantic import BaseModel

from embedding_studio.data_storage.loaders.item_meta import ItemMeta


class DownloadedItem(BaseModel):
    """
    Represents an item that has been downloaded or loaded from a data source.

    DownloadedItem encapsulates the loaded data along with its identifier and metadata,
    providing a standardized structure for data items across different loaders.

    :param id: Unique identifier for the downloaded item
    :param data: The actual data content of the item (can be any type)
    :param meta: Metadata object containing additional information about the item
    """

    id: str
    data: Any
    meta: ItemMeta
