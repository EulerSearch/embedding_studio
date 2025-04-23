from typing import List

import torch
from pydantic import BaseModel, Field


class ImprovementElement(BaseModel):
    """Represents an element in the improvement process with vector representation.

    :param id: Unique identifier for the element
    :param vector: Tensor representation of the element
    :param is_average: List of boolean flags indicating if the vector is an average
    :param user_id: Identifier of the user associated with this element
    """

    id: str
    vector: torch.Tensor
    is_average: List[bool] = Field(default_factory=list)
    user_id: str

    class Config:
        arbitrary_types_allowed = True


class ImprovementInput(BaseModel):
    """Input data structure for the improvement process.

    Contains query and elements that were clicked or not clicked in a session.

    :param session_id: Unique identifier for the session
    :param query: The query element that initiated the session
    :param clicked_elements: List of elements that were clicked by the user
    :param non_clicked_elements: List of elements that were shown but not clicked
    """

    session_id: str
    query: ImprovementElement
    clicked_elements: List[ImprovementElement]
    non_clicked_elements: List[ImprovementElement]

    class Config:
        arbitrary_types_allowed = True
