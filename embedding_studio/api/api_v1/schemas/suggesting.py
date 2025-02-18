from typing import List, Literal

from pydantic import BaseModel, Field


class GetSuggestionsRequest(BaseModel):
    phrase: str = Field(description="The phrase to suggest")
    top_k: int = Field(
        default=3, ge=0, le=100, description="The top k suggestions"
    )


class Span(BaseModel):
    start: int = Field(default=0, description="The start index")
    end: int = Field(default=0, description="The end index")


class Suggestion(BaseModel):
    prefix: str = Field(..., description="The prefix of the suggestion")
    postfix: str = Field(..., description="The postfix of the suggestion")

    matching_span: Span = Field(..., description="The span of the suggestion")

    match_type: Literal[
        "exact", "exact_case_insensitive", "prefix", "fuzzy"
    ] = Field(default="exact", description="The type of match")
    prob: float = Field(
        default=1.0, description="The probability of the suggestion"
    )
    labels: List[str] = Field(
        default_factory=list, description="The labels of the suggestion"
    )


class GetSuggestionsResponse(BaseModel):
    suggestions: List[Suggestion] = Field(
        default_factory=list, description="Suggestions list"
    )


class SuggestingPhrase(BaseModel):
    phrase: str = Field(
        description="Suggesting phrase to be added.",
    )
    labels: List[str] = Field(
        default_factory=list,
        description="Labels related to the suggesting phrase.",
    )
    prob: float = Field(
        default=1.0,
        description="Probability (or weighting) factor for the phrase.",
    )


class SuggestingPhraseWithID(SuggestingPhrase):
    """
    A more complete model for returning phrase details
    alongside its database ID.
    """

    phrase_id: str = Field(description="Unique ID for the suggesting phrase.")


class SuggestingPhrasesAddingRequest(BaseModel):
    phrases: List[SuggestingPhrase] = Field(
        description="Suggesting phrases to be added."
    )


# class SuggestingPhrasesAddingErrors(BaseModel):
#     phrase: str = Field(description="Suggesting phrase.")
#     detail: str = Field(..., description="Error details.")


class SuggestingPhrasesAddingResponse(BaseModel):
    phrase_ids: List[str] = Field(description="Successfully added phrase IDs.")
    # errors: List[SuggestingPhrasesAddingErrors] = Field(
    #     default_factory=list,
    #     description="Any phrases that could not be added, with error details."
    # )


class SuggestingPhrasesAddLabelsRequest(BaseModel):
    phrase_id: str = Field(description="Suggesting phrase ID.")
    labels: List[str] = Field(
        description="Labels to add to the suggesting phrase."
    )


class SuggestingPhrasesRemoveLabelsRequest(BaseModel):
    phrase_id: str = Field(description="Suggesting phrase ID.")
    labels: List[str] = Field(
        description="Labels to remove from the suggesting phrase."
    )


class SuggestingPhrasesRemoveAllLabelsRequest(BaseModel):
    labels: List[str] = Field(
        description="Labels to remove from the suggesting phrase."
    )


class SuggestingPhrasesAdjustProbabilityRequest(BaseModel):
    """
    If your manager method requires a new probability, change 'delta' -> 'prob'.
    If you really intend to increment/decrement the existing probability, keep 'delta'
    as a float. The manager can fetch the current value, add 'delta' to it,
    and persist the new result.
    """

    phrase_id: str = Field(description="Suggesting phrase ID.")
    new_prob: float = Field(description="Value of new probability.")


class SuggestingLabelsResponse(BaseModel):
    labels: List[str] = Field(description="All labels known in the system.")


class GetSuggestionPhraseIDResponse(BaseModel):
    phrase_id: str = Field(description="Suggesting phrase ID.")


class ListPhrasesRequest(BaseModel):
    """
    Paginated request for listing suggesting phrases.
    """

    limit: int = Field(
        ..., description="The maximum number of suggestions to return."
    )
    offset: int = Field(
        default=0, description="The starting index of the suggestions."
    )


class ListPhrasesResponse(BaseModel):
    """
    Paginated response for listing phrases.
    """

    items: List[SuggestingPhraseWithID] = Field(
        default_factory=list,
        description="List of phrases with full details.",
    )
