from typing import List

import torch
from pydantic import BaseModel


class ImprovementElement(BaseModel):
    id: str
    vector: torch.Tensor

    class Config:
        arbitrary_types_allowed = True


class ImprovementInput(BaseModel):
    session_id: str
    query: ImprovementElement
    clicked_elements: List[ImprovementElement]
    non_clicked_elements: List[ImprovementElement]

    class Config:
        arbitrary_types_allowed = True
