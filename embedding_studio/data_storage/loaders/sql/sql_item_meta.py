from typing import Optional

from embedding_studio.data_storage.loaders.item_meta import ItemMeta


class SQLFileMeta(ItemMeta):
    object_id: Optional[int] = None

    @property
    def derived_id(self) -> str:
        """
        Derives a unique identifier based on the row ID.
        Ensures that each row can be uniquely identified, even if `object_id` is not set.
        """
        if self.object_id is not None:
            return f"row:{self.object_id}"
        raise ValueError(
            "Both object_id and derived_id are missing, cannot uniquely identify the row."
        )
