from typing import List, Optional, Union

from pydantic import BaseModel


class MatchQuery(BaseModel):
    field: str
    value: str


class TermQuery(BaseModel):
    field: str
    value: Union[str, int, float, bool]


class TermsQuery(BaseModel):
    field: str
    values: List[Union[str, int, float, bool]]


class MatchPhraseQuery(BaseModel):
    field: str
    value: str


class ExistsQuery(BaseModel):
    field: str


class WildcardQuery(BaseModel):
    field: str
    value: str


class RangeCondition(BaseModel):
    gte: Optional[float] = None
    lte: Optional[float] = None
    gt: Optional[float] = None
    lt: Optional[float] = None
    eq: Optional[float] = None


class RangeQuery(BaseModel):
    field: str
    range: RangeCondition


class BoolQuery(BaseModel):
    must: Optional[
        List[
            Union[
                "MatchQuery",
                "TermQuery",
                "TermsQuery",
                "MatchPhraseQuery",
                "ExistsQuery",
                "WildcardQuery",
                "RangeQuery",
                "BoolQuery",
            ]
        ]
    ] = None
    should: Optional[
        List[
            Union[
                "MatchQuery",
                "TermQuery",
                "TermsQuery",
                "MatchPhraseQuery",
                "ExistsQuery",
                "WildcardQuery",
                "RangeQuery",
                "BoolQuery",
            ]
        ]
    ] = None
    filter: Optional[
        List[
            Union[
                "MatchQuery",
                "TermQuery",
                "TermsQuery",
                "MatchPhraseQuery",
                "ExistsQuery",
                "WildcardQuery",
                "RangeQuery",
                "BoolQuery",
            ]
        ]
    ] = None
    must_not: Optional[
        List[
            Union[
                "MatchQuery",
                "TermQuery",
                "TermsQuery",
                "MatchPhraseQuery",
                "ExistsQuery",
                "WildcardQuery",
                "RangeQuery",
                "BoolQuery",
            ]
        ]
    ] = None

    class Config:
        arbitrary_types_allowed = True


class PayloadFilter(BaseModel):
    query: Union[
        MatchQuery,
        TermQuery,
        TermsQuery,
        MatchPhraseQuery,
        ExistsQuery,
        WildcardQuery,
        RangeQuery,
        BoolQuery,
    ]
