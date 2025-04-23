from pydantic import BaseModel


class QueryItem(BaseModel):
    """
    Base class for query items used in retrieval operations.

    This class extends Pydantic's BaseModel to provide a foundation for
    defining structured query items with validation capabilities.
    """

    class Config:
        arbitrary_types_allowed = True
