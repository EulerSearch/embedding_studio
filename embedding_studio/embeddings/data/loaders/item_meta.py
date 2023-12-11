from abc import abstractmethod

from pydantic import BaseModel


class ItemMeta(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    @property
    @abstractmethod
    def id(self) -> str:
        raise NotImplemented()

    def __hash__(self) -> int:
        # Provide a default_params hash implementation
        return hash(self.id)
