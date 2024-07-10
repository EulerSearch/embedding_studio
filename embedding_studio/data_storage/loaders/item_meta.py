from abc import abstractmethod
from typing import Any, Dict, Optional

from pydantic import BaseModel


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
