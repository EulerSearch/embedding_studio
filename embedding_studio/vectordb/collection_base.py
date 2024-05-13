from embedding_studio.models.embeddings.collections import CollectionInfo
from embedding_studio.vectordb.collection import Collection


class CollectionBase(Collection):
    def __init__(self, collection_info: CollectionInfo):
        self.collection_info = collection_info

    def get_info(self) -> CollectionInfo:
        return self.collection_info
