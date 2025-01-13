from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel


class ObjectPart(BaseModel):
    vector: List[float]
    part_id: Optional[str] = None


class ObjectСommonData(BaseModel):
    object_id: str
    payload: Optional[Dict[str, Any]] = None
    storage_meta: Dict[str, Any]

    user_id: Optional[str] = None
    session_id: Optional[str] = None
    original_id: Optional[str] = None


class Object(ObjectСommonData):
    parts: List[ObjectPart]


class ObjectsCommonDataBatch(BaseModel):
    objects_info: List[ObjectСommonData]
    total: int
    next_offset: Optional[int] = None


class FoundObject(BaseModel):
    object_id: str
    parts_found: int
    payload: Optional[Dict[str, Any]] = None
    storage_meta: Dict[str, Any]


class SimilarObject(FoundObject):
    distance: float  # aggregated


class SearchResults(BaseModel):
    found_objects: List[Union[FoundObject, SimilarObject]]
    next_offset: Optional[Union[str, int]] = None
