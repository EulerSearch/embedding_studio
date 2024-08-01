from sqlalchemy import Numeric, and_, cast, func, not_, or_, text

from embedding_studio.models.payload.models import (
    BoolQuery,
    ExistsQuery,
    MatchPhraseQuery,
    MatchQuery,
    PayloadFilter,
    RangeQuery,
    TermQuery,
    TermsQuery,
    WildcardQuery,
)


def translate_query_to_orm_filters(
    payload_filter: PayloadFilter,
    prefix: str = "payload",
    language: str = "simple",
) -> list:
    """Translate PayloadQuery to SQLAlchemy filter conditions.
    :param payload_filter: PayloadQuery to translate
    :param prefix: The prefix used for JSONB fields in the PostgreSQL query. Defaults to "payload".
    :param language: The language of the payload query. Defaults to "simple".
    :return: A list of SQLAlchemy filter conditions.
    """
    filters = []

    query = payload_filter.query
    if isinstance(query, MatchQuery):
        filters.append(
            func.to_tsvector(
                language,
                func.jsonb_extract_path_text(text(prefix), query.field),
            ).match(query.value)
        )

    elif isinstance(query, TermQuery):
        filters.append(
            cast(text(f"{prefix} ->> '{query.field}'"), Numeric) == query.value
        )

    elif isinstance(query, TermsQuery):
        filters.append(
            cast(text(f"{prefix} ->> '{query.field}'"), Numeric).in_(
                query.values
            )
        )

    elif isinstance(query, MatchPhraseQuery):
        filters.append(
            func.to_tsvector(
                language,
                func.jsonb_extract_path_text(text(prefix), query.field),
            ).match(query.value)
        )

    elif isinstance(query, ExistsQuery):
        filters.append(text(f"{prefix} ? '{query.field}'"))

    elif isinstance(query, WildcardQuery):
        filters.append(
            func.to_tsvector(
                language,
                func.jsonb_extract_path_text(text(prefix), query.field),
            ).match(query.value.replace("*", ":*"))
        )

    elif isinstance(query, RangeQuery):
        range_cond = query.range
        field_value = cast(text(f"{prefix} ->> '{query.field}'"), Numeric)
        if range_cond.gte is not None:
            filters.append(field_value >= range_cond.gte)
        if range_cond.lte is not None:
            filters.append(field_value <= range_cond.lte)
        if range_cond.gt is not None:
            filters.append(field_value > range_cond.gt)
        if range_cond.lt is not None:
            filters.append(field_value < range_cond.lt)
        if range_cond.eq is not None:
            filters.append(field_value == range_cond.eq)

    elif isinstance(query, BoolQuery):
        bool_conditions = []
        if query.must:
            must_conditions = [
                translate_query_to_orm_filters(
                    PayloadFilter(must_query), prefix
                )
                for must_query in query.must
            ]
            bool_conditions.append(
                and_(
                    *[item for sublist in must_conditions for item in sublist]
                )
            )
        if query.should:
            should_conditions = [
                translate_query_to_orm_filters(
                    PayloadFilter(should_query), prefix
                )
                for should_query in query.should
            ]
            bool_conditions.append(
                or_(
                    *[
                        item
                        for sublist in should_conditions
                        for item in sublist
                    ]
                )
            )
        if query.filter:
            filter_conditions = [
                translate_query_to_orm_filters(
                    PayloadFilter(filter_query), prefix
                )
                for filter_query in query.filter
            ]
            bool_conditions.append(
                and_(
                    *[
                        item
                        for sublist in filter_conditions
                        for item in sublist
                    ]
                )
            )
        if query.must_not:
            must_not_conditions = [
                translate_query_to_orm_filters(
                    PayloadFilter(must_not_query), prefix
                )
                for must_not_query in query.must_not
            ]
            bool_conditions.append(
                not_(
                    and_(
                        *[
                            item
                            for sublist in must_not_conditions
                            for item in sublist
                        ]
                    )
                )
            )
        filters.append(and_(*bool_conditions))

    return filters
