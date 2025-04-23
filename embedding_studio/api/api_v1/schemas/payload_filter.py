from typing import List, Optional, Union

from pydantic import BaseModel, Field


class QueryBase(BaseModel):
    """
    Base class for all query types with a flag to control
    whether the query field is in the payload or is a direct column.
    """

    force_not_payload: bool = Field(default=False)


class SingleValueQuery(QueryBase):
    """
    Represents a query that matches a single value for
    a field, supporting various data types.
    """

    field: str
    value: Union[str, int, float, bool]


class MultipleValuesQuery(QueryBase):
    """
    Used for queries that check against multiple values for a field.
    """

    field: str
    values: List[Union[str, int, float, bool]]


class SingleTextValueQuery(QueryBase):
    """
    Specialized version of SingleValueQuery for text-only fields.
    """

    field: str
    value: str


class MultipleTextValuesQuery(QueryBase):
    """
    Specialized version of MultipleValuesQuery for text-only fields.
    """

    field: str
    values: List[str]


class MatchQuery(BaseModel):
    """
    Performs text matching using PostgreSQL's text search capabilities.
    """

    match: SingleTextValueQuery


class TermQuery(BaseModel):
    """
    Performs exact matching for a field against a single value.
    """

    term: SingleValueQuery


class TermsQuery(BaseModel):
    """
    Matches fields against multiple values (equivalent to SQL's IN operator).
    """

    terms: MultipleValuesQuery


class ListHasAnyQuery(BaseModel):
    """
    Checks if an array field contains any of the provided values.
    """

    any: MultipleValuesQuery


class ListHasAllQuery(BaseModel):
    """
    Checks if an array field contains all of the provided values.
    """

    all: MultipleValuesQuery


class MatchPhraseQuery(BaseModel):
    """
    Similar to MatchQuery but specifically for matching phrases.
    """

    match_phrase: SingleTextValueQuery


class ExistsQuery(QueryBase):
    """
    Checks if a field exists in the data.
    """

    field: str


class WildcardQuery(BaseModel):
    """
    Supports pattern matching with wildcards.
    """

    wildcard: SingleTextValueQuery


class RangeCondition(BaseModel):
    """
    Defines conditions for range queries (greater than, less than, etc.).
    """

    gte: Optional[float] = None
    lte: Optional[float] = None
    gt: Optional[float] = None
    lt: Optional[float] = None
    eq: Optional[float] = None


class RangeQuery(QueryBase):
    """
    Queries for fields with values in a specified numeric range.
    """

    field: str
    range: RangeCondition


class BoolQuery(BaseModel):
    """
    Combines multiple query conditions with boolean logic (must, should, filter, must_not).
    """

    must: Optional[
        List[
            Union[
                "MatchQuery",
                "TermQuery",
                "TermsQuery",
                "ListHasAllQuery",
                "ListHasAnyQuery",
                "MatchPhraseQuery",
                "WildcardQuery",
                "RangeQuery",
                "BoolQuery",
                "ExistsQuery",
            ]
        ]
    ] = None
    should: Optional[
        List[
            Union[
                "MatchQuery",
                "TermQuery",
                "TermsQuery",
                "ListHasAllQuery",
                "ListHasAnyQuery",
                "MatchPhraseQuery",
                "WildcardQuery",
                "RangeQuery",
                "BoolQuery",
                "ExistsQuery",
            ]
        ]
    ] = None
    filter: Optional[
        List[
            Union[
                "MatchQuery",
                "TermQuery",
                "TermsQuery",
                "ListHasAllQuery",
                "ListHasAnyQuery",
                "MatchPhraseQuery",
                "WildcardQuery",
                "RangeQuery",
                "BoolQuery",
                "ExistsQuery",
            ]
        ]
    ] = None
    must_not: Optional[
        List[
            Union[
                "MatchQuery",
                "TermQuery",
                "TermsQuery",
                "ListHasAllQuery",
                "ListHasAnyQuery",
                "MatchPhraseQuery",
                "WildcardQuery",
                "RangeQuery",
                "BoolQuery",
                "ExistsQuery",
            ]
        ]
    ] = None

    class Config:
        arbitrary_types_allowed = True


class PayloadFilter(BaseModel):
    """ """

    query: Union[
        MatchQuery,
        TermQuery,
        TermsQuery,
        MatchPhraseQuery,
        ListHasAllQuery,
        ListHasAnyQuery,
        ExistsQuery,
        WildcardQuery,
        RangeQuery,
        BoolQuery,
    ]
