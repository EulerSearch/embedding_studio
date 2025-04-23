from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


class ObjectPart(BaseModel):
    """
    ObjectPart: Represents a single vector part of an object, containing the vector
    itself, part ID, and a flag indicating if it's an average vector. Enables
    storing multiple vectors per object for more nuanced representation.
    """

    vector: Optional[List[float]] = Field(None)
    part_id: Optional[str] = None
    is_average: Optional[bool] = Field(default=False)


class ObjectСommonData(BaseModel):
    """
    ObjectСommonData: Contains common metadata for objects including ID, payload,
    storage metadata, and optional user/session information. Provides the core
    identity and context information for stored objects.
    """

    object_id: str
    payload: Optional[Dict[str, Any]] = None
    storage_meta: Dict[str, Any]

    user_id: Optional[str] = None
    session_id: Optional[str] = None
    original_id: Optional[str] = None


class Object(ObjectСommonData):
    """
    Object: Extends ObjectСommonData to include a list of ObjectPart vectors.
    Represents a complete storable entity with both metadata and vector embeddings.
    """

    parts: List[ObjectPart]


class ObjectWithDistance(Object):
    """
    ObjectWithDistance: Extends Object to include a distance score from a query
    vector. Used for returning similarity search results with their relevance scores.
    """

    distance: Optional[float] = None


class ObjectsCommonDataBatch(BaseModel):
    """
    ObjectsCommonDataBatch: A container for batches of objects with pagination
    information (total count and next offset). Facilitates efficient batch
    processing and pagination of large result sets.
    """

    objects_info: List[ObjectСommonData]
    total: int
    next_offset: Optional[int] = None


class FoundObject(BaseModel):
    """
    FoundObject: A simplified object representation for search results, containing
    ID, number of matching parts, payload, and storage metadata. Used for returning
    minimal necessary information in search results.
    """

    object_id: str
    parts_found: int
    payload: Optional[Dict[str, Any]] = None
    storage_meta: Dict[str, Any]


class SimilarObject(FoundObject):
    """
    SimilarObject: Extends FoundObject to include a distance score. Used
    specifically for similarity search results to provide relevance information.
    """

    distance: float  # aggregated


class SearchResults(BaseModel):
    """
    SearchResults: A container for search results, including found objects,
    pagination information, and optional metadata. Serves as the standard response
    format for search operations.
    """

    found_objects: List[Union[FoundObject, SimilarObject]]
    next_offset: Optional[Union[str, int]] = None
    total_count: Optional[int] = None
    meta_info: Optional[Any] = None
