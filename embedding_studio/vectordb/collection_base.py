from embedding_studio.models.embeddings.collections import CollectionInfo
from embedding_studio.vectordb.collection import Collection


class CollectionBase(Collection):
    """
    A base implementation of the Collection interface.

    This class provides a basic implementation for the Collection interface
    with common functionality.

    :param collection_info: Information about the collection
    """

    def __init__(self, collection_info: CollectionInfo):
        self.collection_info = collection_info

    def get_info(self) -> CollectionInfo:
        """
        Get information about the collection.

        :return: Collection information object
        """
        return self.collection_info
