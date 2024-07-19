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
        """
        raise NotImplemented()

    @property
    def id(self) -> str:
        """
        Returns a unique identifier for an item. Defaults to `object_id` if available,
        otherwise uses `derived_id` provided by the subclass.

        The `id` property serves as a universal getter for an item's unique identifier, abstracting away
        the details of how this identifier is constructed, whether directly provided or derived.
        This design allows easy storage, retrieval, and comparison of items in data structures or when logging,
        regardless of the item's specific type or source.
        """
        return self.object_id or self.derived_id

    def __hash__(self) -> int:
        # Provide a default hash implementation based on the unique id
        return hash(self.id)
