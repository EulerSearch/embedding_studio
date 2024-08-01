from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel


class ObjectPart(BaseModel):
    vector: List[float]
    part_id: Optional[str] = None


class Object(BaseModel):
    object_id: str
    parts: List[ObjectPart]
    payload: Optional[Dict[str, Any]] = None


class FoundObject(BaseModel):
    object_id: str
    parts_found: int
    payload: Optional[Dict[str, Any]] = None


class SimilarObject(FoundObject):
    distance: float  # aggregated


class SearchResults(BaseModel):
    found_objects: List[Union[FoundObject, SimilarObject]]
    next_offset: Optional[Union[str, int]] = None
