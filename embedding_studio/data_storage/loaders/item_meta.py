from abc import abstractmethod
from typing import Any, Dict, Optional

from pydantic import BaseModel

# object_id
# What it is: An optional attribute that can be set for an instance of ItemMeta.
# Why it's needed: It is used as a unique identifier for the item if set. This can be a predefined or external identifier (e.g., UUID, database ID, etc.).
# When it is used: When there is a direct and explicit way to uniquely identify the item.
#
# derived_id
# What it is: An abstract property that subclasses of ItemMeta must implement.
# Why it's needed: Allows creating a unique identifier based on the item's properties if object_id is not set. This ensures that each item has a unique identifier.
# When it is used: When object_id is absent and an identifier needs to be created based on other properties (e.g., file path or bucket name).
#
# id
# What it is: A property that returns either object_id or derived_id.
# Why it's needed: A universal way to get a unique identifier for the item, whether it is directly set or calculated.
# When it is used: Always, when a unique identifier for the item is needed.
#
# Hash
# Why it's needed: So that instances of ItemMeta can be used as keys in dictionaries or stored in sets. The hash is calculated based on the id, ensuring uniqueness.
# When it is used: When storing or comparing multiple items, allowing it to be done efficiently and correctly.


class ItemMeta(BaseModel):
    """
    Base class for metadata about items in the data storage system.

    ItemMeta serves as a foundation for tracking and identifying data items.
    It provides mechanisms for unique identification through either explicit IDs
    or derived identifiers based on item properties.

    :param object_id: Optional explicit identifier for the item
    :param payload: Optional dictionary containing additional metadata
    """

    object_id: Optional[str] = None
    payload: Optional[Dict[str, Any]] = None

    class Config:
        arbitrary_types_allowed = True

    @property
    @abstractmethod
    def derived_id(self) -> str:
        """
        Abstract property that must be implemented by subclasses.

        The purpose of `derived_id` is to provide a unique identifier that is specific to the subclass's context.
        This is particularly useful in scenarios where `object_id` is not set, ensuring that every instance
        can still be uniquely identified based on its inherent attributes (like file path, database ID, etc.).

        This mechanism supports polymorphic behavior where different types of items might have different
        methods of constructing a unique ID, which is not possible to standardize at the `ItemMeta` level.

        Example implementation:
        ```python
        @property
        def derived_id(self) -> str:
            return f"{self.file_path}:{self.creation_timestamp}"
        ```
        """
        raise NotImplemented()

    @property
    def id(self) -> str:
        """
        Returns a unique identifier for an item. Defaults to `object_id` if available,
        otherwise uses `derived_id` provided by the subclass.

        The `id` property serves as a universal getter for an item's unique identifier, abstracting away
        the details of how this identifier is constructed, whether directly provided or derived.
        This design allows easy items_set, retrieval, and comparison of items in data structures or when logging,
        regardless of the item's specific type or source.
        """
        return self.object_id or self.derived_id

    def __hash__(self) -> int:
        """
        Provides a hash implementation based on the unique id.

        This enables instances to be used as keys in dictionaries or stored in sets.

        :return: An integer hash value based on the item's id
        """
        # Provide a default hash implementation based on the unique id
        return hash(self.id)


class ItemMetaWithSourceInfo(ItemMeta):
    """
    Extended ItemMeta that includes information about the source of the item.

    This class adds a source identifier to the metadata, which is incorporated into
    the derived ID to ensure uniqueness across different sources.

    :param source_name: Name of the source from which the item originates
    """

    source_name: str

    @property
    def derived_id(self) -> str:
        """
        Creates a derived ID that includes the source name.

        Extends the base class derived_id implementation by prefixing it with the source name.

        :return: A string representing a unique identifier that includes source information

        Example usage:
        ```python
        class FileItemMeta(ItemMetaWithSourceInfo):
            file_path: str

            @property
            def derived_id(self) -> str:
                # This will call ItemMetaWithSourceInfo.derived_id
                # which will prefix with source_name
                return f"{self.file_path}"
        ```
        """
        return f"{self.source_name}:{super().derived_id}"
