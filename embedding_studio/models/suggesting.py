from typing import List, Literal, Optional

from bson import ObjectId
from pydantic import BaseModel, Field, field_validator

from embedding_studio.db.common import PyObjectId


class SuggestingPhrase(BaseModel):
    phrase: str = Field(...)
    labels: List[str] = Field(default_factory=list)
    prob: float = Field(default=1.0)

    @field_validator("prob")
    @classmethod
    def check_prob(cls, prob):
        if not (0.0 <= prob <= 1.0):
            raise ValueError("prob must be between 0.0 and 1.0")
        return prob


class Chunk(BaseModel):
    value: str = Field(...)
    search_field: List[str] = Field(default_factory=list)


class SearchDocument(BaseModel):
    id: Optional[PyObjectId] = Field(default=ObjectId, alias="_id")
    phrase: str = Field(...)
    labels: List[str] = Field(default_factory=list)
    prob: float = Field(default=1.0)

    chunks: List[Chunk] = Field(default_factory=list)

    # Use the built-in model_dump but transform it to your desired format
    def get_flattened_dict(self) -> dict:
        base = self.model_dump()  # This gives you the nested structure

        # Convert to flattened format
        flattened = {
            "id": str(base["id"]),
            "phrase": base["phrase"],
            "n_chunks": len(base["chunks"]),
            "prob": base["prob"],
            "labels": base["labels"],
        }

        for i, chunk in enumerate(base["chunks"]):
            flattened[f"chunk_{i}"] = chunk["value"]
            flattened[f"search_{i}"] = chunk["search_field"]

        return flattened

    @classmethod
    def from_flattened_dict(cls, data: dict) -> "SearchDocument":
        """Create SearchDocument from flattened dictionary format"""
        n_chunks = data["n_chunks"]
        chunks = []

        for i in range(n_chunks):
            chunk = Chunk(
                value=data[f"chunk_{i}"], search_field=data[f"search_{i}"]
            )
            chunks.append(chunk)

        return cls(
            id=PyObjectId(data["id"]),
            chunks=chunks,
            prob=data.get("prob", 1.0),
            labels=data.get("labels", []),
        )


class Span(BaseModel):
    start: int = Field(default=0, ge=0)
    end: int = Field(..., ge=0)

    @field_validator("end")
    @classmethod
    def check_start_end(cls, end, info):
        start = info.data.get("start", 0)
        if start >= end:
            raise ValueError("start must be less than end")
        return end


class SuggestingRequestChunks(BaseModel):
    found_chunks: List[str] = Field(default_factory=list)
    next_chunk: str = Field(...)


class SuggestingRequestSpans(BaseModel):
    found_chunk_spans: List[Span] = Field(default_factory=list)
    next_chunk_span: Optional[Span] = Field(default=None)


class SuggestingRequest(BaseModel):
    chunks: SuggestingRequestChunks = Field(...)
    spans: SuggestingRequestSpans = Field(...)


class Suggest(BaseModel):
    chunks: List[str] = Field(default_factory=list)
    prefix_chunks: List[str] = Field(default_factory=list)
    match_type: Literal[
        "exact", "exact_case_insensitive", "prefix", "fuzzy"
    ] = Field(default="exact")
    prob: float = Field(default=1.0)
    labels: List[str] = Field(default_factory=list)

    @field_validator("prob")
    @classmethod
    def check_prob(cls, prob):
        if not (0.0 <= prob <= 1.0):
            raise ValueError("prob must be between 0.0 and 1.0")
        return prob
