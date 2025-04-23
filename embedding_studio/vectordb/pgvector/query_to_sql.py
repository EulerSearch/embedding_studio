from typing import Tuple

from sqlalchemy import Float, Integer, Text, and_, cast, func, not_, or_, text
from sqlalchemy.types import Numeric as NumericType

from embedding_studio.models.payload.models import (
    BoolQuery,
    ExistsQuery,
    ListHasAllQuery,
    ListHasAnyQuery,
    MatchPhraseQuery,
    MatchQuery,
    PayloadFilter,
    RangeQuery,
    TermQuery,
    TermsQuery,
    WildcardQuery,
)


def get_cast_type(value):
    """
    Determine the appropriate SQLAlchemy type based on Python value.

    :param value: The Python value to determine type for
    :return: SQLAlchemy type class (Text, Integer, Float, etc.)
    """
    if isinstance(value, str):
        return Text
    elif isinstance(value, int):
        return Integer
    elif isinstance(value, float):
        return Float
    # Add more rules if needed, or fallback to Text:
    return Text


def group_values_by_type(values):
    """
    Group values by their Python-derived SQLAlchemy type.

    Organizes a list of values into groups by their corresponding SQLAlchemy types.

    :param values: List of values to group
    :return: Dictionary mapping SQLAlchemy types to lists of values
    """
    from collections import defaultdict

    grouped = defaultdict(list)
    for v in values:
        col_type = get_cast_type(v)
        grouped[col_type].append(v)
    return grouped


def translate_query_to_orm_filters(
    payload_filter: PayloadFilter,
    prefix: str = "payload",
    language: str = "simple",
) -> Tuple[list, list]:
    """
    Translate a PayloadFilter into SQLAlchemy filter conditions.

    Converts a PayloadFilter object into SQLAlchemy ORM filter conditions
    that can be used in SQLAlchemy queries.

    :param payload_filter: PayloadFilter to translate
    :param prefix: Prefix for JSON field references (default: "payload")
    :param language: Text search language for text search operations (default: "simple")
    :return: Tuple of (filters, solid_filters) where:
            - filters: List of SQLAlchemy filter conditions for the final combined condition
            - solid_filters: List of SQLAlchemy filter conditions for more specialized contexts
    """

    filters = []
    solid_filters = []

    query = payload_filter.query

    # -----------------------------------------------------
    # MatchQuery
    # -----------------------------------------------------
    if isinstance(query, MatchQuery):
        if query.match.force_not_payload:
            # If the field is NOT in JSON, treat it as a raw column
            filters.append(
                func.to_tsvector(
                    language,
                    getattr(query.match, "field"),
                ).match(query.match.value)
            )
            solid_filters.append(
                func.to_tsvector(
                    language,
                    getattr(query.match, "field"),
                ).match(query.match.value)
            )
        else:
            # Normal JSON-based approach
            filters.append(
                func.to_tsvector(
                    language,
                    func.jsonb_extract_path_text(
                        text(prefix), query.match.field
                    ),
                ).match(query.match.value)
            )
            solid_filters.append(
                func.to_tsvector(
                    language,
                    func.jsonb_extract_path_text(
                        text(prefix), query.match.field
                    ),
                ).match(query.match.value)
            )

    # -----------------------------------------------------
    # TermQuery
    # -----------------------------------------------------
    elif isinstance(query, TermQuery):
        value = (
            f"'{query.term.value}'"
            if isinstance(query.term.value, str)
            else query.term.value
        )
        if query.term.force_not_payload:
            # Field is a raw column
            # e.g. text("my_column = 'some_value'")
            filters.append(text(f"{query.term.field} = {value}"))
            solid_filters.append(text(f"{query.term.field} = {value}"))
        else:
            # Field is in JSON
            filters.append(
                text(f"{prefix} ->> '{query.term.field}' = {value}")
            )
            solid_filters.append(
                text(f"{prefix} ->> '{query.term.field}' = {value}")
            )

    # -----------------------------------------------------
    # TermsQuery
    # -----------------------------------------------------
    elif isinstance(query, TermsQuery):
        values_str = ", ".join(
            (f"'{v}'" if isinstance(v, str) else str(v))
            for v in query.terms.values
        )
        if query.terms.force_not_payload:
            # Field is a raw column
            filters.append(text(f"{query.terms.field} IN ({values_str})"))
            solid_filters.append(
                text(f"{query.terms.field} IN ({values_str})")
            )
        else:
            # Field is in JSON
            filters.append(
                text(f"{prefix} ->> '{query.terms.field}' IN ({values_str})")
            )
            solid_filters.append(
                text(f"{prefix} ->> '{query.terms.field}' IN ({values_str})")
            )

    # -----------------------------------------------------
    # ListHasAllQuery
    # -----------------------------------------------------
    elif isinstance(query, ListHasAllQuery):
        values_str = ", ".join(
            (f"'{v}'" if isinstance(v, str) else str(v))
            for v in query.all.values
        )
        if query.all.force_not_payload:
            # Typically doesn't make sense to do ?& array[...] if not JSON,
            # but we'll assume the user wants the same logic on a real column.
            # If you truly have a text[] or similar, you'd adapt accordingly.
            filters.append(text(f"{query.all.field} ?& array[{values_str}]"))
            solid_filters.append(
                text(f"{query.all.field} ?& array[{values_str}]")
            )
        else:
            filters.append(
                text(f"{prefix} -> '{query.all.field}' ?& array[{values_str}]")
            )
            solid_filters.append(
                text(f"{prefix} -> '{query.all.field}' ?& array[{values_str}]")
            )

    # -----------------------------------------------------
    # ListHasAnyQuery
    # -----------------------------------------------------
    elif isinstance(query, ListHasAnyQuery):
        values_str = ", ".join(
            (f"'{v}'" if isinstance(v, str) else str(v))
            for v in query.any.values
        )
        if query.any.force_not_payload:
            filters.append(text(f"{query.any.field} ?| array[{values_str}]"))
            solid_filters.append(
                text(f"{query.any.field} ?| array[{values_str}]")
            )
        else:
            filters.append(
                text(f"{prefix} -> '{query.any.field}' ?| array[{values_str}]")
            )
            solid_filters.append(
                text(f"{prefix} -> '{query.any.field}' ?| array[{values_str}]")
            )

    # -----------------------------------------------------
    # MatchPhraseQuery
    # -----------------------------------------------------
    elif isinstance(query, MatchPhraseQuery):
        if query.match_phrase.force_not_payload:
            filters.append(
                func.to_tsvector(
                    language,
                    getattr(query.match_phrase, "field"),
                ).match(query.match_phrase.value)
            )
            solid_filters.append(
                func.to_tsvector(
                    language,
                    getattr(query.match_phrase, "field"),
                ).match(query.match_phrase.value)
            )
        else:
            filters.append(
                func.to_tsvector(
                    language,
                    func.jsonb_extract_path_text(
                        text(prefix), query.match_phrase.field
                    ),
                ).match(query.match_phrase.value)
            )
            solid_filters.append(
                func.to_tsvector(
                    language,
                    func.jsonb_extract_path_text(
                        text(prefix), query.match_phrase.field
                    ),
                ).match(query.match_phrase.value)
            )

    # -----------------------------------------------------
    # ExistsQuery
    # -----------------------------------------------------
    elif isinstance(query, ExistsQuery):
        if query.force_not_payload:
            # If not in JSON, "EXISTS" logically might mean "column is not null" or similar
            # We'll do a direct test for NOT NULL. Adjust if needed.
            filters.append(text(f"{query.field} IS NOT NULL"))
            solid_filters.append(text(f"{query.field} IS NOT NULL"))
        else:
            filters.append(text(f"{prefix} ? '{query.field}'"))
            solid_filters.append(text(f"{prefix} ? '{query.field}'"))

    # -----------------------------------------------------
    # WildcardQuery
    # -----------------------------------------------------
    elif isinstance(query, WildcardQuery):
        tsquery_value = query.wildcard.value.replace("*", ":*")
        if query.wildcard.force_not_payload:
            filters.append(
                func.to_tsvector(
                    language,
                    getattr(query.wildcard, "field"),
                ).match(tsquery_value)
            )
            solid_filters.append(
                func.to_tsvector(
                    language,
                    getattr(query.wildcard, "field"),
                ).match(tsquery_value)
            )
        else:
            filters.append(
                func.to_tsvector(
                    language,
                    func.jsonb_extract_path_text(
                        text(prefix), query.wildcard.field
                    ),
                ).match(tsquery_value)
            )
            solid_filters.append(
                func.to_tsvector(
                    language,
                    func.jsonb_extract_path_text(
                        text(prefix), query.wildcard.field
                    ),
                ).match(tsquery_value)
            )

    # -----------------------------------------------------
    # RangeQuery
    # -----------------------------------------------------
    elif isinstance(query, RangeQuery):
        range_cond = query.range
        if query.force_not_payload:
            # Field is real column => cast if numeric is needed
            # Or just treat it as numeric
            field_col = cast(text(query.field), NumericType)
        else:
            field_col = cast(
                text(f"{prefix} ->> '{query.field}'"), NumericType
            )

        if range_cond.gte is not None:
            filters.append(field_col >= range_cond.gte)
            solid_filters.append(field_col >= range_cond.gte)
        if range_cond.lte is not None:
            filters.append(field_col <= range_cond.lte)
            solid_filters.append(field_col <= range_cond.lte)
        if range_cond.gt is not None:
            filters.append(field_col > range_cond.gt)
            solid_filters.append(field_col > range_cond.gt)
        if range_cond.lt is not None:
            filters.append(field_col < range_cond.lt)
            solid_filters.append(field_col < range_cond.lt)
        if range_cond.eq is not None:
            filters.append(field_col == range_cond.eq)
            solid_filters.append(field_col == range_cond.eq)

    # -----------------------------------------------------
    # BoolQuery
    # -----------------------------------------------------
    elif isinstance(query, BoolQuery):
        # Force_not_payload doesn't typically apply to the container query
        # but might apply to sub-queries individually.
        bool_conditions = []

        if query.must:
            must_conditions = [
                translate_query_to_orm_filters(
                    PayloadFilter(query=must_query), prefix, language
                )
                for must_query in query.must
            ]
            bool_conditions.append(
                and_(
                    *[
                        item
                        for sublist in must_conditions
                        for item in sublist[0]
                    ]
                )
            )
            solid_filters.append(
                and_(
                    *[
                        item
                        for sublist in must_conditions
                        for item in sublist[0]
                    ]
                )
            )

        if query.should:
            should_conditions = [
                translate_query_to_orm_filters(
                    PayloadFilter(query=should_query), prefix, language
                )
                for should_query in query.should
            ]
            bool_conditions.append(
                or_(
                    *[
                        item
                        for sublist in should_conditions
                        for item in sublist[0]
                    ]
                )
            )
            solid_filters += [
                item for sublist in should_conditions for item in sublist[1]
            ]

        if query.filter:
            filter_conditions = [
                translate_query_to_orm_filters(
                    PayloadFilter(query=filter_query), prefix, language
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
            solid_filters.append(
                and_(
                    *[
                        item
                        for sublist in filter_conditions
                        for item in sublist[0]
                    ]
                )
            )

        if query.must_not:
            must_not_conditions = [
                translate_query_to_orm_filters(
                    PayloadFilter(query=must_not_query), prefix, language
                )
                for must_not_query in query.must_not
            ]
            bool_conditions.append(
                not_(
                    and_(
                        *[
                            item
                            for sublist in must_not_conditions
                            for item in sublist[0]
                        ]
                    )
                )
            )
            solid_filters.append(
                not_(
                    and_(
                        *[
                            item
                            for sublist in must_not_conditions
                            for item in sublist[0]
                        ]
                    )
                )
            )

        if bool_conditions:
            filters.append(and_(*bool_conditions))

    return filters, solid_filters


def translate_query_to_sql_filters(
    payload_filter: PayloadFilter,
    prefix: str = "payload",
    language: str = "simple",
) -> str:
    """
    Translate a PayloadFilter to a raw SQL WHERE-clause string.

    Converts a PayloadFilter object into a raw SQL string that can be used
    in a WHERE clause. Handles various query types including match, term, range,
    exists, and boolean queries.

    :param payload_filter: PayloadFilter to translate
    :param prefix: Prefix for JSON field references (default: "payload")
    :param language: Text search language for text search operations (default: "simple")
    :return: SQL WHERE clause string representing the filter conditions
    """
    query = payload_filter.query
    # -----------------------------------------------------
    # MatchQuery
    # -----------------------------------------------------
    if isinstance(query, MatchQuery):
        if query.match.force_not_payload:
            # direct column
            return (
                f"to_tsvector('{language}', {query.match.field}) "
                f"@@ to_tsquery('{language}', '{query.match.value}')"
            )
        else:
            return (
                f"to_tsvector('{language}', jsonb_extract_path_text({prefix}, '{query.match.field}')) "
                f"@@ to_tsquery('{language}', '{query.match.value}')"
            )

    # -----------------------------------------------------
    # TermQuery
    # -----------------------------------------------------
    elif isinstance(query, TermQuery):
        value = query.term.value
        field = query.term.field

        if query.term.force_not_payload:
            # direct column
            if isinstance(value, (int, float)):
                return f"({field})::numeric = {value}"
            elif isinstance(value, bool):
                return f"({field})::boolean = {str(value).lower()}"
            else:
                return f"{field} = '{value}'"
        else:
            if isinstance(value, (int, float)):
                return f"(({prefix} ->> '{field}')::numeric) = {value}"
            elif isinstance(value, bool):
                return f"(({prefix} ->> '{field}')::boolean) = {str(value).lower()}"
            else:
                return f"({prefix} ->> '{field}') = '{value}'"

    # -----------------------------------------------------
    # TermsQuery
    # -----------------------------------------------------
    elif isinstance(query, TermsQuery):
        values = query.terms.values
        field = query.terms.field

        if not values:
            return "TRUE"  # No values => no filtering

        first_val = values[0]

        if query.terms.force_not_payload:
            # direct column
            if isinstance(first_val, (int, float)):
                values_str = ", ".join(str(v) for v in values)
                return f"(({field})::numeric) IN ({values_str})"
            elif isinstance(first_val, bool):
                values_str = ", ".join(str(v).lower() for v in values)
                return f"(({field})::boolean) IN ({values_str})"
            else:
                values_str = ", ".join(f"'{v}'" for v in values)
                return f"{field} IN ({values_str})"
        else:
            # JSON
            if isinstance(first_val, (int, float)):
                values_str = ", ".join(str(v) for v in values)
                return f"(({prefix} ->> '{field}')::numeric) IN ({values_str})"
            elif isinstance(first_val, bool):
                values_str = ", ".join(str(v).lower() for v in values)
                return f"(({prefix} ->> '{field}')::boolean) IN ({values_str})"
            else:
                values_str = ", ".join(f"'{v}'" for v in values)
                return f"({prefix} ->> '{field}') IN ({values_str})"

    # -----------------------------------------------------
    # ListHasAllQuery
    # -----------------------------------------------------
    elif isinstance(query, ListHasAllQuery):
        values = query.all.values
        field = query.all.field
        if not values:
            return "TRUE"
        values_str = ", ".join(f"'{v}'" for v in values)

        if query.allforce_not_payload:
            return f"{field} ?& array[{values_str}]"
        else:
            return f"{prefix} -> '{field}' ?& array[{values_str}]"

    # -----------------------------------------------------
    # ListHasAnyQuery
    # -----------------------------------------------------
    elif isinstance(query, ListHasAnyQuery):
        values = query.any.values
        field = query.any.field
        if not values:
            return "TRUE"
        values_str = ", ".join(f"'{v}'" for v in values)

        if query.any.force_not_payload:
            return f"{field} ?| array[{values_str}]"
        else:
            return f"{prefix} -> '{field}' ?| array[{values_str}]"

    # -----------------------------------------------------
    # MatchPhraseQuery
    # -----------------------------------------------------
    elif isinstance(query, MatchPhraseQuery):
        if query.match_phrase.force_not_payload:
            return (
                f"to_tsvector('{language}', {query.match_phrase.field}) "
                f"@@ phraseto_tsquery('{language}', '{query.match_phrase.value}')"
            )
        else:
            return (
                f"to_tsvector('{language}', jsonb_extract_path_text({prefix}, '{query.match_phrase.field}')) "
                f"@@ phraseto_tsquery('{language}', '{query.match_phrase.value}')"
            )

    # -----------------------------------------------------
    # ExistsQuery
    # -----------------------------------------------------
    elif isinstance(query, ExistsQuery):
        field = query.field
        if query.force_not_payload:
            # interpret "exists" as "field IS NOT NULL"
            return f"{field} IS NOT NULL"
        else:
            return f"{prefix} ? '{field}'"

    # -----------------------------------------------------
    # WildcardQuery
    # -----------------------------------------------------
    elif isinstance(query, WildcardQuery):
        tsquery_value = query.wildcard.value.replace("*", ":*")
        field = query.wildcard.field
        if query.wildcard.force_not_payload:
            return (
                f"to_tsvector('{language}', {field}) "
                f"@@ to_tsquery('{language}', '{tsquery_value}')"
            )
        else:
            return (
                f"to_tsvector('{language}', jsonb_extract_path_text({prefix}, '{field}')) "
                f"@@ to_tsquery('{language}', '{tsquery_value}')"
            )

    # -----------------------------------------------------
    # RangeQuery
    # -----------------------------------------------------
    elif isinstance(query, RangeQuery):
        field = query.field
        range_cond = query.range
        parts = []

        # If forced, interpret as real column
        if query.force_not_payload:
            base = f"({field})::numeric"
        else:
            base = f"(({prefix} ->> '{field}')::numeric)"

        if range_cond.gte is not None:
            parts.append(f"{base} >= {range_cond.gte}")
        if range_cond.lte is not None:
            parts.append(f"{base} <= {range_cond.lte}")
        if range_cond.gt is not None:
            parts.append(f"{base} > {range_cond.gt}")
        if range_cond.lt is not None:
            parts.append(f"{base} < {range_cond.lt}")
        if range_cond.eq is not None:
            parts.append(f"{base} = {range_cond.eq}")

        return " AND ".join(parts) if parts else "TRUE"

    # -----------------------------------------------------
    # BoolQuery
    # -----------------------------------------------------
    elif isinstance(query, BoolQuery):
        conditions = []

        # must => AND
        if query.must:
            must_clauses = []
            for must_item in query.must:
                sub_sql = translate_query_to_sql_filters(
                    PayloadFilter(query=must_item), prefix, language
                )
                must_clauses.append(f"({sub_sql})")

            if must_clauses:
                conditions.append(" AND ".join(must_clauses))

        # should => OR
        if query.should:
            should_clauses = []
            for should_item in query.should:
                sub_sql = translate_query_to_sql_filters(
                    PayloadFilter(query=should_item), prefix, language
                )
                should_clauses.append(f"({sub_sql})")
            if should_clauses:
                conditions.append(" OR ".join(should_clauses))

        # filter => AND
        if query.filter:
            filter_clauses = []
            for filter_item in query.filter:
                sub_sql = translate_query_to_sql_filters(
                    PayloadFilter(query=filter_item), prefix, language
                )
                filter_clauses.append(f"({sub_sql})")
            if filter_clauses:
                conditions.append(" AND ".join(filter_clauses))

        # must_not => NOT ( AND(...) )
        if query.must_not:
            must_not_clauses = []
            for must_not_item in query.must_not:
                sub_sql = translate_query_to_sql_filters(
                    PayloadFilter(query=must_not_item), prefix, language
                )
                must_not_clauses.append(f"({sub_sql})")
            if must_not_clauses:
                conditions.append(f"NOT ( {' AND '.join(must_not_clauses)} )")

        # If nothing was specified, default to "TRUE"
        if not conditions:
            return "TRUE"

        # By default, we AND everything at the top level
        return " AND ".join(f"({c})" for c in conditions)

    # -----------------------------------------------------
    # Fallback
    # -----------------------------------------------------
    else:
        return "TRUE"
