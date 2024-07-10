from typing import Any

from pydantic import BaseModel

from embedding_studio.data_storage.loaders.item_meta import ItemMeta


class DownloadedItem(BaseModel):
    id: str
    data: Any
    meta: ItemMeta
