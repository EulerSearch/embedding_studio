from embedding_studio.data_storage.loaders.item_meta import (
    ItemMetaWithSourceInfo,
)


class PgsqlItemMetaWithSourceInfo(ItemMetaWithSourceInfo):
    """
    Metadata for a PostgreSQL item including its source name and ID.
    Used to trace and identify objects across multiple data sources.
    """

    @property
    def derived_id(self) -> str:
        """
        Return unique ID combining source name and object ID.
        Format: source_name:object_id
        """
        return f"{self.source_name}:{self.object_id}"
