from abc import ABC, abstractmethod

from embedding_studio.vectordb.collection import Collection


class Optimization(ABC):
    """
    Abstract base class for defining optimization strategies for vector collections.

    Optimization strategies can be applied to collections to improve performance,
    indexing quality, or other operational aspects.

    :param name: The name of the optimization strategy
    """

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def __call__(self, collection: Collection):
        """
        Apply the optimization strategy to a collection.

        This method is called when the optimization is applied to a collection.

        :param collection: The collection to optimize
        :return: None

        Example implementation:
        ```python
        def __call__(self, collection: Collection):
            # Example: Create an index for faster similarity search
            if not collection.get_state_info().index_created:
                collection.create_index()
                print(f"Applied {self.name} optimization to {collection.get_info().collection_id}")
        ```
        """
