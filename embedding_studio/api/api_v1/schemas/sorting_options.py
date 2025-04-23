from typing import Literal

from pydantic import BaseModel, Field


class SortByOptions(BaseModel):
    """
    A model that defines how results should be sorted, specifying the field name,
    sort direction (ascending or descending), and whether the field is outside the payload JSON.
    """

    field: str
    order: Literal["asc", "desc"] = "asc"
    force_not_payload: bool = Field(default=False)
