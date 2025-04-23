from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class GetSuggestionsRequest(BaseModel):
    """
    Request model for obtaining autocompletion suggestions based on user input.
    Captures the partial phrase being typed along with any domain context.
    Allows configuration of result count through top_k parameter.
    Essential for powering interactive search and input autocompletion features.
    """

    phrase: str = Field(description="The phrase to suggest")
    domain: Optional[str] = Field(default=None, description="Expected domain")
    top_k: int = Field(
        default=3, ge=0, le=100, description="The top k suggestions"
    )


class Span(BaseModel):
    """
    Represents a text position range with start and end indices.
    Used to precisely identify the location of matches within input text.
    Enables accurate highlighting and text replacement in user interfaces.
    Critical for maintaining context when inserting suggested text.
    """

    start: int = Field(default=0, description="The start index")
    end: int = Field(default=0, description="The end index")


class Suggestion(BaseModel):
    """
    Complete representation of a text autocompletion suggestion.
    Contains both prefix (matched text) and postfix (suggested completion).
    Includes metadata about match quality, probability and categorization.
    Provides all information needed to present and apply suggestions to user input.
    """

    prefix: str = Field(..., description="The prefix of the suggestion")
    postfix: str = Field(..., description="The postfix of the suggestion")
    matching_span: Span = Field(..., description="The span of the suggestion")

    match_type: Literal[
        "exact", "exact_case_insensitive", "prefix", "fuzzy", "unknown"
    ] = Field(default="exact", description="The type of match")
    prob: float = Field(
        default=1.0, description="The probability of the suggestion"
    )
    labels: List[str] = Field(
        default_factory=list, description="The labels of the suggestion"
    )
    domains: List[str] = Field(
        default_factory=list, description="The domains of the suggestion"
    )


class GetSuggestionsResponse(BaseModel):
    """
    Container for multiple suggestion results returned from the suggestion engine.
    Delivers ordered list of possible completions for the user's partial input.
    Forms the core response payload for the suggestion API endpoints.
    Enables UIs to present multiple autocompletion options to users.
    """

    suggestions: List[Suggestion] = Field(
        default_factory=list, description="Suggestions list"
    )


class SuggestingPhrase(BaseModel):
    """
    Represents a phrase that can be suggested to users during text input.
    Contains the core phrase text along with categorization metadata.
    Includes probability weighting to influence suggestion ranking.
    Forms the basic unit of managed content in the suggestion system.
    """

    phrase: str = Field(
        description="Suggesting phrase to be added.",
    )
    labels: List[str] = Field(
        default_factory=list,
        description="Labels related to the suggesting phrase.",
    )
    domains: List[str] = Field(
        default_factory=list,
        description="Domains related to the suggesting phrase.",
    )
    prob: float = Field(
        default=1.0,
        description="Probability (or weighting) factor for the phrase.",
    )


class SuggestingPhrasesAddingRequest(BaseModel):
    """
    Request for adding multiple phrases to the suggestion system at once.
    Supports batch insertion of new autocompletion candidates with full metadata.
    Enables efficient bulk population and updates of the suggestion database.
    Critical for maintaining fresh and relevant suggestion content.
    """

    phrases: List[SuggestingPhrase] = Field(
        description="Suggesting phrases to be added."
    )


class SuggestingPhrasesAddingResponse(BaseModel):
    """
    Response confirming successful phrase additions to the suggestion system.
    Returns identifiers for all successfully added phrases for tracking.
    Enables clients to reference added phrases in subsequent operations.
    Provides confirmation of successful database updates.
    """

    phrase_ids: List[str] = Field(description="Successfully added phrase IDs.")


class SuggestingPhrasesAddLabelsRequest(BaseModel):
    """
    Request to associate new category labels with an existing suggestion phrase.
    Enables taxonomy expansion and content categorization after initial creation.
    Supports evolution of phrase classification without requiring recreation.
    Essential for maintaining organized and filterable suggestion content.
    """

    phrase_id: str = Field(description="Suggesting phrase ID.")
    labels: List[str] = Field(
        description="Labels to add to the suggesting phrase."
    )


class SuggestingPhrasesRemoveLabelsRequest(BaseModel):
    """
    Request to disassociate specific category labels from a suggestion phrase.
    Enables refinement of phrase classification and taxonomy maintenance.
    Supports content organization workflows with minimal data disruption.
    Provides granular control over suggestion metadata management.
    """

    phrase_id: str = Field(description="Suggesting phrase ID.")
    labels: List[str] = Field(
        description="Labels to remove from the suggesting phrase."
    )


class SuggestingPhrasesRemoveAllLabelsRequest(BaseModel):
    """
    Request to completely remove specific labels from all phrases in the system.
    Supports sweeping taxonomy changes and global classification cleanup.
    Enables efficient handling of deprecated or obsolete categorization terms.
    Critical for maintaining consistent and clean category systems.
    """

    labels: List[str] = Field(
        description="Labels to remove from the suggesting phrase."
    )


class SuggestingPhrasesAddDomainsRequest(BaseModel):
    """
    Request to associate specific domains with an existing suggestion phrase.
    Enables contextual scoping of suggestions to particular application areas.
    Supports multi-tenant or multi-domain suggestion deployments.
    Ensures suggestions appear only in relevant application contexts.
    """

    phrase_id: str = Field(description="Suggesting phrase ID.")
    domains: List[str] = Field(
        description="Domains to add to the suggesting phrase."
    )


class SuggestingPhrasesRemoveDomainsRequest(BaseModel):
    """
    Request to disassociate specific domains from a suggestion phrase.
    Enables refinement of phrase visibility and contextual applicability.
    Supports targeted cleanup of domain-specific suggestion content.
    Provides precise control over suggestion availability by context.
    """

    phrase_id: str = Field(description="Suggesting phrase ID.")
    domains: List[str] = Field(
        description="Domains to remove from the suggesting phrase."
    )


class SuggestingPhrasesRemoveAllDomainsRequest(BaseModel):
    """
    Request to completely remove specific domains from all phrases in the system.
    Supports global domain cleanup when contexts are deprecated or removed.
    Enables efficient maintenance of domain-specific suggestion partitioning.
    Critical for managing multi-tenant suggestion data integrity.
    """

    domains: List[str] = Field(
        description="Labels to remove from the suggesting phrase."
    )


class SuggestingPhrasesAdjustProbabilityRequest(BaseModel):
    """
    Request to modify the ranking probability of a specific suggestion phrase.
    Enables fine-tuning of suggestion ordering and prominence.
    Supports manual optimization of suggestion quality based on usage patterns.
    Critical for curating high-quality suggestion experiences.
    """

    phrase_id: str = Field(description="Suggesting phrase ID.")
    new_prob: float = Field(description="Value of new probability.")


class SuggestingLabelsResponse(BaseModel):
    """
    Response listing all category labels known to the suggestion system.
    Provides visibility into the full taxonomy of available classifications.
    Enables discovery of available filtering and categorization options.
    Supports taxonomy-aware UIs and administration interfaces.
    """

    labels: List[str] = Field(description="All labels known in the system.")


class GetSuggestionPhraseIDResponse(BaseModel):
    """
    Response providing the unique identifier for a specific suggestion phrase.
    Enables referencing phrases by stable IDs for subsequent operations.
    Bridges between human-readable phrases and system identifiers.
    Supports workflows that require phrase identification before modification.
    """

    phrase_id: str = Field(description="Suggesting phrase ID.")


class ListPhrasesRequest(BaseModel):
    """
    Paginated request for retrieving batches of suggestion phrases.
    Enables efficient browsing and discovery of available suggestions.
    Supports administrative interfaces and content management tools.
    Critical for systems with large suggestion databases.
    """

    limit: int = Field(
        ..., description="The maximum number of suggestions to return."
    )
    offset: int = Field(
        default=0, description="The starting index of the suggestions."
    )


class ListPhrasesResponse(BaseModel):
    """
    Paginated response containing batches of complete suggestion phrases.
    Returns full phrase objects with all associated metadata.
    Enables rich display and management of suggestion content.
    Supports efficient exploration of large suggestion databases.
    """

    items: List[SuggestingPhrase] = Field(
        default_factory=list,
        description="List of phrases with full details.",
    )
