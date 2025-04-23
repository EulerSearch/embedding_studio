from typing import Optional

from embedding_studio.data_storage.loaders.item_meta import ItemMeta


class SQLFileMeta(ItemMeta):
    """
    Metadata for items stored in SQL databases.

    This class extends ItemMeta to represent metadata for rows
    in SQL database tables, providing a unique identifier for each row.

    Attributes:
        object_id: Optional identifier for the SQL database row.
    """

    object_id: Optional[str] = None

    @property
    def derived_id(self) -> str:
        """
        Derives a unique identifier based on the row ID.

        This method creates a unique identifier string based on the object_id,
        prefixing it with 'row:' to indicate it's a database row identifier.

        Ensures that each row can be uniquely identified, even if `object_id` is not set.

        :return: A string representing the unique identifier for the row.
        :raises ValueError: If object_id is not set and no alternative ID is available.
        """
        if self.object_id is not None:
            return f"row:{self.object_id}"
        raise ValueError(
            "Both object_id and derived_id are missing, cannot uniquely identify the row."
        )
