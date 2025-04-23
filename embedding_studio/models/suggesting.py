import hashlib
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator

from embedding_studio.utils.redis_utils import ft_escape_punctuation


class SuggestingPhrase(BaseModel):
    """
    Represents a phrase that can be suggested to users as they type.
    It includes the actual text, categorizing labels, domain information,
    and a probability score that helps rank suggestions.
    """

    phrase: str = Field(...)
    labels: List[str] = Field(default_factory=list)
    domains: List[str] = Field(default_factory=list)
    prob: float = Field(default=1.0)

    @field_validator("prob")
    @classmethod
    def check_prob(cls, prob):
        if not (0.0 <= prob <= 1.0):
            raise ValueError("prob must be between 0.0 and 1.0")
        return prob


class Chunk(BaseModel):
    """
    A simple container for a text fragment or piece of a phrase.
    Used to break down phrases into smaller, manageable parts that
    can be matched against user input.
    """

    value: str = Field(...)


class SearchDocument(BaseModel):
    """
    A comprehensive representation of a suggestion phrase with additional metadata.
    It stores the complete phrase, its component chunks, probability, labels, domains,
    and can convert between database storage format and application format.
    """

    phrase: str = Field(...)
    labels: List[str] = Field(default_factory=list)
    prob: float = Field(default=1.0)
    chunks: List[Chunk] = Field(default_factory=list)
    domains: List[str] = Field(default_factory=list)
    is_original_phrase: Optional[bool] = Field(default=False)

    class Config:
        populate_by_name = True

    @staticmethod
    def hash_string(text):
        """Create a hash of a string value for faster indexing"""
        return hashlib.md5(text.encode()).hexdigest()

    # Use the built-in model_dump but transform it to your desired format
    def get_flattened_dict(self) -> dict:
        base = self.model_dump()  # This gives you the nested structure

        label_ids = []
        labels = []
        for label in base["labels"]:
            label_id = hashlib.md5(label.encode()).hexdigest()
            label_ids.append(label_id)

            labels.append(label)

        domain_tags = []
        for domain in base["domains"]:
            domain_tags.append(f"__{domain.replace('-', '_')}__")

        # Convert to flattened format
        flattened = {
            "id": SearchDocument.hash_string(base["phrase"]),
            "phrase": base["phrase"],
            "n_chunks": len(base["chunks"]),
            "prob": base["prob"],
            "labels": " ".join(labels),
            "label_ids": " ".join(label_ids),
            "domains": " ".join(domain_tags),
            "is_original_phrase": int(base["is_original_phrase"]),
        }

        for i, chunk in enumerate(base["chunks"]):
            flattened[f"chunk_{i}"] = ft_escape_punctuation(
                chunk["value"].lower()
            )

        return flattened

    @classmethod
    def from_flattened_dict(cls, data: dict) -> "SearchDocument":
        """Create SearchDocument from flattened dictionary format"""
        n_chunks = data["n_chunks"]
        chunks = []

        for i in range(n_chunks):
            chunk = Chunk(value=data[f"chunk_{i}"])
            chunks.append(chunk)

        domains = []
        for value in data.get("domains", "").split(" "):
            domains.append(value[2:-2])

        labels = []
        for value in data.get("labels", "").split(" "):
            labels.append(value)

        return cls(
            phrase=data["phrase"],
            chunks=chunks,
            prob=data.get("prob", 1.0),
            labels=labels,
            domains=domains,
            is_original_phrase=bool(data.get("is_original_phrase", False)),
        )


class Span(BaseModel):
    """
    Tracks the position of text within a larger string using start and end indices.
    This helps the system know exactly where certain words or phrases appear
    within user input, making it easier to match and highlight suggestions.
    """

    start: int = Field(default=0, ge=0)
    end: int = Field(..., ge=0)

    @field_validator("end")
    @classmethod
    def check_start_end(cls, end, info):
        start = info.data.get("start", 0)
        if start >= end:
            raise ValueError("start must be less than end")
        return end

    def __len__(self) -> int:
        return self.end - self.start


class SuggestingRequestChunks(BaseModel):
    """
    Tracks what parts of a query have already been processed and what comes next.
    This helps the suggestion system understand what the user has already typed
    and what suggestions would make sense to offer next.
    """

    found_chunks: List[str] = Field(default_factory=list)
    next_chunk: str = Field(default="")


class SuggestingRequestSpans(BaseModel):
    """
    Similar to SuggestingRequestChunks but includes position information.
    This provides more precise tracking of where text appears in the input,
    enhancing the suggestion system's ability to provide contextual completions.
    """

    found_chunk_spans: List[Span] = Field(default_factory=list)
    next_chunk_span: Optional[Span] = Field(default=None)


class SuggestingRequest(BaseModel):
    """
    A complete request for suggestions containing both the text chunks and
    their positions. This model combines everything the suggestion system
    needs to understand what the user has typed and what domain they're in.
    """

    chunks: SuggestingRequestChunks = Field(...)
    spans: SuggestingRequestSpans = Field(...)
    domain: Optional[str] = Field(default=None)


class Suggest(BaseModel):
    """
    An actual suggestion that can be shown to the user. It includes the full
    suggested text, how it matches the user's input, probability score, and
    relevant labels for categorizing and filtering.
    """

    chunks: List[str] = Field(default_factory=list)
    prefix_chunks: List[str] = Field(default_factory=list)
    match_type: Literal[
        "exact", "exact_case_insensitive", "prefix", "fuzzy", "unknown"
    ] = Field(default="exact")
    prob: float = Field(default=1.0)
    labels: List[str] = Field(default_factory=list)

    @field_validator("prob")
    @classmethod
    def check_prob(cls, prob):
        if not (0.0 <= prob <= 1.0):
            raise ValueError("prob must be between 0.0 and 1.0")
        return prob
