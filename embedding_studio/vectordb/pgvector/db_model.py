import json
import logging
from typing import Any, Dict, List, Optional, Tuple, Type

import numpy as np
import sqlalchemy
from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Boolean,
    ForeignKey,
    Index,
    String,
    and_,
    delete,
    insert,
    select,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, insert as pg_insert
from sqlalchemy.ext.declarative import declarative_base, declared_attr
from sqlalchemy.orm import mapped_column, relationship
from sqlalchemy.sql import func

from embedding_studio.models.embeddings.collections import CollectionInfo
from embedding_studio.models.embeddings.models import (
    MetricAggregationType,
    MetricType,
)
from embedding_studio.models.embeddings.objects import (
    FoundObject,
    Object,
    ObjectPart,
    ObjectWithDistance,
    ObjectСommonData,
    SimilarObject,
)
from embedding_studio.models.payload.models import PayloadFilter
from embedding_studio.models.sort_by.models import SortByOptions
from embedding_studio.vectordb.pgvector.query_to_sql import (
    translate_query_to_orm_filters,
    translate_query_to_sql_filters,
)

logger = logging.getLogger(__name__)


class DimensionsMismatch(Exception):
    pass


Base = declarative_base()


def convert_vectors(vectors_data: str) -> np.array:
    """
    Convert string representation of vectors to numpy array.

    :param vectors_data: String representation of vectors from database
    :return: Numpy array of vectors
    """
    return np.array(
        [json.loads(arr) for arr in vectors_data[2:-2].split('","')]
    )


class DbObjectBase(Base):
    """
    Abstract base class for vector database object tables.

    This class defines the common schema for objects stored in the vector database.
    It uses SQLAlchemy's declarative base and mapped columns for database mapping.
    """

    __abstract__ = True

    @declared_attr
    def object_id(cls):
        """
        Primary key column for the object.

        :return: SQLAlchemy Column definition
        """
        return mapped_column(String(128), primary_key=True)

    @declared_attr
    def payload(cls):
        """
        JSONB column for storing arbitrary payload data.

        :return: SQLAlchemy Column definition
        """
        return mapped_column(JSONB)

    @declared_attr
    def storage_meta(cls):
        """
        JSONB column for storing metadata about the object storage.

        :return: SQLAlchemy Column definition
        """
        return mapped_column(JSONB)

    @declared_attr
    def original_id(cls):
        """
        String column for storing the original ID if this is a derived object.

        :return: SQLAlchemy Column definition
        """
        return mapped_column(String(128))

    @declared_attr
    def user_id(cls):
        """
        String column for storing the user ID associated with the object.

        :return: SQLAlchemy Column definition
        """
        return mapped_column(String(128))

    @declared_attr
    def session_id(cls):
        """
        String column for storing the session ID associated with the object.

        :return: SQLAlchemy Column definition
        """
        return mapped_column(String(128))


class DbObjectPartBase(Base):
    """
    Abstract base class for vector database object part tables.

    This class defines the common schema for object parts (vectors) stored in the database.
    Each object can have multiple parts, each with its own vector.
    """

    __abstract__ = True

    @declared_attr
    def part_id(cls):
        """
        Primary key column for the object part.

        :return: SQLAlchemy Column definition
        """
        return mapped_column(String(128), primary_key=True)

    @declared_attr
    def object_id(cls):
        """
        Foreign key column referencing the parent object.

        :return: SQLAlchemy Column definition
        """
        return mapped_column(
            String(128),
            ForeignKey(f"dbo_{cls.__tablename__[5:]}.object_id"),
            index=True,
        )

    @declared_attr
    def vector(cls):
        """
        Vector column for storing the embedding vector.

        :return: SQLAlchemy Column definition
        """
        return mapped_column(Vector)

    @declared_attr
    def is_average(cls):
        """
        Boolean column indicating if this part represents an average vector.

        :return: SQLAlchemy Column definition
        """
        return mapped_column(Boolean)

    @declared_attr
    def user_id(cls):
        """
        String column for storing the user ID associated with the part.

        :return: SQLAlchemy Column definition
        """
        return mapped_column(String(128))


class DbObjectImpl:
    """
    Implementation mixin for database object tables.

    This class provides common methods for working with object tables,
    including SQL statement generation and data conversion utilities.
    """

    @classmethod
    def create_table(cls, pg_database: sqlalchemy.Engine):
        """
        Create the database table for objects.

        :param pg_database: SQLAlchemy engine
        """
        cls.__table__.create(pg_database, checkfirst=True)

    @classmethod
    def insert_objects_statement(cls, db_objects: List["DbObject"]):
        """
        Generate a SQL statement for inserting objects.

        :param db_objects: List of DbObject instances to insert
        :return: SQLAlchemy insert statement
        """
        db_dicts = cls.db_objects_to_dicts(db_objects)
        return insert(cls).values(db_dicts)

    @classmethod
    def upsert_objects_statement(cls, db_objects: List["DbObject"]):
        """
        Generate a SQL statement for upserting objects.

        :param db_objects: List of DbObject instances to upsert
        :return: SQLAlchemy upsert statement
        """
        db_dicts = cls.db_objects_to_dicts(db_objects)
        insert_st = pg_insert(cls).values(db_dicts)
        update_dict = {
            "payload": insert_st.excluded.payload,
            "storage_meta": insert_st.excluded.storage_meta,
        }
        return insert_st.on_conflict_do_update(
            index_elements=[cls.object_id], set_=update_dict
        )

    @classmethod
    def db_object_to_dict(cls, db_object: "DbObject") -> Dict[str, Any]:
        """
        Convert a DbObject instance to a dictionary.

        :param db_object: DbObject instance
        :return: Dictionary representation of the object
        """
        return {
            "object_id": db_object.object_id,
            "payload": db_object.payload,
            "storage_meta": db_object.storage_meta,
            "original_id": db_object.original_id,
            "user_id": db_object.user_id,
            "session_id": db_object.session_id,
        }

    @classmethod
    def db_objects_to_dicts(
        cls, db_objects: List["DbObject"]
    ) -> List[Dict[str, Any]]:
        """
        Convert a list of DbObject instances to a list of dictionaries.

        :param db_objects: List of DbObject instances
        :return: List of dictionaries
        """
        return [cls.db_object_to_dict(obj) for obj in db_objects]

    @classmethod
    def delete_statement(cls, object_ids: List[str]):
        """
        Generate a SQL statement for deleting objects.

        :param object_ids: List of object IDs to delete
        :return: SQLAlchemy delete statement
        """
        return delete(cls).where(cls.object_id.in_(object_ids))

    @classmethod
    def get_total_statement(cls, originals_only: bool = True):
        """
        Generate a SQL statement for counting objects.

        :param originals_only: If True, count only original objects
        :return: SQLAlchemy select statement
        """
        query = select(func.count(cls.object_id))

        if originals_only:
            # Add condition to count only original objects
            query = query.where(cls.original_id.is_(None))

        return query

    @classmethod
    def get_objects_common_data_batch_statement(
        cls,
        limit: int,
        offset: Optional[int] = None,
        originals_only: bool = True,
    ):
        """
        Generate a SQL statement for retrieving common data for a batch of objects.

        :param limit: Maximum number of objects to retrieve
        :param offset: Number of objects to skip
        :param originals_only: If True, retrieve only original objects
        :return: SQLAlchemy select statement
        """
        # Base query
        query = select(cls.object_id, cls.payload, cls.storage_meta)

        # Add condition to retrieve only original objects if specified
        if originals_only:
            query = query.where(cls.original_id.is_(None))

        # Apply limit and offset for batching
        query = query.limit(limit)
        if offset is not None:
            query = query.offset(offset)

        return query

    @classmethod
    def objects_common_data_from_db(cls, rows) -> List[ObjectСommonData]:
        """
        Convert database rows to ObjectСommonData instances.

        :param rows: Database result rows
        :return: List of ObjectСommonData instances
        """
        return [
            ObjectСommonData(
                object_id=row.object_id,
                payload=row.payload,
                storage_meta=row.storage_meta,
            )
            for row in rows
        ]


class DbObjectPartImpl:
    """
    Implementation mixin for database object part tables.

    This class provides common methods for working with object part tables,
    including SQL statement generation for vector operations and data conversion utilities.
    """

    search_index = None
    db_object_class = None

    @classmethod
    def initialize(cls, search_index, db_object_class):
        """
        Initialize the implementation with a search index and object class.

        :param search_index: Search index configuration
        :param db_object_class: DbObject class
        """
        cls.search_index = search_index
        cls.db_object_class = db_object_class

    @classmethod
    def create_table(cls, pg_database: sqlalchemy.Engine):
        """
        Create the database table for object parts.

        :param pg_database: SQLAlchemy engine
        """
        cls.__table__.create(pg_database, checkfirst=True)

    @classmethod
    def validate_dimensions(cls, vector: List[float]):
        """
        Validate that a vector has the correct dimensions.

        :param vector: Vector to validate
        :raises DimensionsMismatch: If dimensions don't match
        """
        dim = len(vector)
        if dim != cls.search_index.dimensions:
            raise DimensionsMismatch(
                f"Dimensions mismatch: input vector({dim}), expected vector({cls.search_index.dimensions})"
            )

    @classmethod
    def distance_expression(cls, vector: List[float]):
        """
        Generate a SQLAlchemy expression for calculating vector distance.

        The specific distance metric used depends on the search index configuration.

        :param vector: Vector to compare against
        :return: SQLAlchemy expression for vector distance
        """
        cls.validate_dimensions(vector)
        if cls.search_index.metric_type is MetricType.COSINE:
            return cls.vector.cosine_distance(vector)
        if cls.search_index.metric_type is MetricType.DOT:
            return cls.vector.max_inner_product(vector)
        if cls.search_index.metric_type is MetricType.EUCLID:
            return cls.vector.l2_distance(vector)
        raise RuntimeError(
            f"unknown metric type: {cls.search_index.metric_type.value}"
        )

    @classmethod
    def aggregated_distance_expression(cls, vector: List[float]):
        """
        Generate a SQLAlchemy expression for calculating aggregated vector distance.

        The specific aggregation method depends on the search index configuration.

        :param vector: Vector to compare against
        :return: SQLAlchemy expression for aggregated vector distance
        """
        dst = cls.distance_expression(vector)
        if (
            cls.search_index.metric_aggregation_type
            is MetricAggregationType.AVG
        ):
            return func.avg(dst)
        elif (
            cls.search_index.metric_aggregation_type
            is MetricAggregationType.MIN
        ):
            return func.min(dst)
        raise RuntimeError(
            f"unknown metric aggregation type: {cls.search_index.metric_aggregation_type.value}"
        )

    @classmethod
    def find_by_id_statement(cls, object_ids: List[float]):
        """
        Generate a SQL statement for finding object parts by object IDs.

        :param object_ids: List of object IDs
        :return: SQLAlchemy select statement
        """
        return (
            select(
                cls.db_object_class.object_id,
                cls.part_id,
                cls.vector,
                cls.is_average,
                cls.db_object_class.payload,
                cls.db_object_class.storage_meta,
                cls.db_object_class.original_id,
                cls.db_object_class.user_id,
                cls.db_object_class.session_id,
            )
            .join(cls.db_object_class)
            .where(cls.db_object_class.object_id.in_(object_ids))
        )

    @classmethod
    def find_by_original_id_statement(cls, object_ids: List[float]):
        """
        Generate a SQL statement for finding object parts by original object IDs.

        :param object_ids: List of original object IDs
        :return: SQLAlchemy select statement
        """
        return (
            select(
                cls.db_object_class.object_id,
                cls.part_id,
                cls.vector,
                cls.db_object_class.payload,
                cls.db_object_class.storage_meta,
                cls.db_object_class.original_id,
                cls.db_object_class.user_id,
                cls.db_object_class.session_id,
            )
            .join(cls.db_object_class)
            .where(cls.db_object_class.original_id.in_(object_ids))
        )

    @classmethod
    def similarity_search_statement(
        cls,
        query_vector: List[float],
        limit: int,
        offset: Optional[int],
        max_distance: Optional[float],
        payload_filter: Optional[PayloadFilter],
        sort_by: Optional[SortByOptions] = None,
        user_id: Optional[str] = None,
        with_vectors: bool = False,
        similarity_first: bool = False,
        meta_info: Any = None,
    ):
        """
        Generate a SQL statement for similarity search.

        Creates a text-based SQL statement that calls the appropriate vector search function
        based on the provided parameters.

        :param query_vector: Vector to compare against
        :param limit: Maximum number of results
        :param offset: Number of results to skip
        :param max_distance: Maximum distance threshold
        :param payload_filter: Filter for payload
        :param sort_by: Sorting options
        :param user_id: Filter by user ID
        :param with_vectors: Include vectors in results
        :param similarity_first: Sort by similarity first
        :param meta_info: Additional metadata
        :return: SQLAlchemy text statement
        """
        collection_id = cls.__name__.replace("DbObjectPart_", "")
        metric_type = (
            cls.search_index.metric_type.value.lower()
        )  # Using value property to match the function naming
        input_vector = json.dumps(query_vector)
        offset_value = offset if offset is not None else 0
        max_distance_text = (
            str(max_distance) if max_distance is not None else "null"
        )
        user_id_text = f"'{user_id}'" if user_id else "null"

        # Default enlarged limits (for prefetching)
        enlarged_limit = limit
        enlarged_offset = offset
        if meta_info and isinstance(meta_info, dict):
            enlarged_limit = meta_info.get("enlarged_limit", limit)
            enlarged_offset = meta_info.get("enlarged_offset", offset)

        average_only = (
            "TRUE"
            if cls.search_index.metric_aggregation_type
            == MetricAggregationType.AVG
            else "FALSE"
        )

        # Build payload filter SQL
        payload_filter_sql = (
            f"$FILTER${translate_query_to_sql_filters(payload_filter)}$FILTER$"
            if payload_filter
            else "null"
        )

        # Determine if we're using sort fields
        if sort_by and not similarity_first:
            sort_field_text = f"'{sort_by.field}'"
            sort_order_text = (
                "'desc'" if sort_by.order.lower() == "desc" else "'asc'"
            )
            is_payload_text = (
                "TRUE" if not sort_by.force_not_payload else "FALSE"
            )
        else:
            sort_field_text = "null"
            sort_order_text = "'asc'"
            is_payload_text = "FALSE"

        # Determine which function to use
        use_advanced = payload_filter is not None or user_id is not None
        similarity_ordered = sort_by is None or similarity_first

        # Build function name
        if use_advanced:
            if similarity_ordered:
                function_prefix = (
                    "advanced_so" if not with_vectors else "advanced_v_so"
                )
            else:
                function_prefix = (
                    "advanced" if not with_vectors else "advanced_v"
                )
        else:
            if similarity_ordered:
                function_prefix = (
                    "simple_so" if not with_vectors else "simple_v_so"
                )
            else:
                function_prefix = "simple" if not with_vectors else "simple_v"

        function_name = f"{function_prefix}_{collection_id}_{metric_type}"

        # Prepare SQL query based on chosen function
        if similarity_ordered:
            # Similarity ordered functions have different parameters
            if use_advanced:
                sql = f"""
    SELECT
        result_object_id as object_id,
        result_payload as payload,
        result_storage_meta as storage_meta,
        result_user_id as user_id,
        result_original_id as original_id,
        result_part_ids as part_ids,
        {'result_vectors as vectors,' if with_vectors else ''}
        result_distance as distance,
        subset_count
    FROM {function_name}(
        '{input_vector}'::vector,
        {user_id_text},
        {payload_filter_sql},
        {limit},
        {offset_value},
        {max_distance_text},
        {enlarged_limit},
        {enlarged_offset},
        {average_only}
    );"""
            else:
                # Simple similarity ordered functions don't have subset_count and user_id/filter parameters
                sql = f"""
    SELECT
        result_object_id as object_id,
        result_payload as payload,
        result_storage_meta as storage_meta,
        result_user_id as user_id,
        result_original_id as original_id,
        result_part_ids as part_ids,
        {'result_vectors as vectors,' if with_vectors else ''}
        result_distance as distance,
        subset_count
    FROM {function_name}(
        '{input_vector}'::vector,
        {limit},
        {offset_value},
        {max_distance_text},
        {enlarged_limit},
        {enlarged_offset},
        {average_only}
    );"""
        else:
            # Non-similarity ordered functions
            if use_advanced:
                sql = f"""
    SELECT
        result_object_id as object_id,
        result_payload as payload,
        result_storage_meta as storage_meta,
        result_user_id as user_id,
        result_original_id as original_id,
        result_part_ids as part_ids,
        {'result_vectors as vectors,' if with_vectors else ''}
        result_distance as distance,
        subset_count
    FROM {function_name}(
        '{input_vector}'::vector,
        {user_id_text},
        {payload_filter_sql},
        {limit},
        {offset_value},
        {max_distance_text},
        {sort_field_text},
        {sort_order_text},
        {is_payload_text},
        {enlarged_limit},
        {enlarged_offset},
        {average_only}
    );"""
            else:
                # Simple non-similarity ordered functions
                sql = f"""
    SELECT
        result_object_id as object_id,
        result_payload as payload,
        result_storage_meta as storage_meta,
        result_user_id as user_id,
        result_original_id as original_id,
        result_part_ids as part_ids,
        {'result_vectors as vectors,' if with_vectors else ''}
        result_distance as distance,
        subset_count
    FROM {function_name}(
        '{input_vector}'::vector,
        {limit},
        {offset_value},
        {max_distance_text},
        {sort_field_text},
        {sort_order_text},
        {is_payload_text},
        {enlarged_limit},
        {enlarged_offset},
        {average_only}
    );"""

        return text(sql)

    @classmethod
    def payload_count_statement(cls, payload_filter: PayloadFilter):
        """
        Generate a SQL statement for counting objects matching a payload filter.

        :param payload_filter: Filter for payload
        :return: SQLAlchemy select statement
        """
        if payload_filter:
            payload_conditions, _ = translate_query_to_orm_filters(
                payload_filter, prefix="payload"
            )
        else:
            payload_conditions = []

        payload_conditions.append(cls.db_object_class.original_id.is_(None))

        count_query = select(
            func.count(cls.db_object_class.object_id),
        ).where(and_(*payload_conditions))

        return count_query

    @classmethod
    def payload_search_statement(
        cls,
        payload_filter: PayloadFilter,
        limit: int,
        offset: Optional[int] = None,
        sort_by: Optional[SortByOptions] = None,
    ):
        """
        Generate a SQL statement for searching objects by payload filter.

        :param payload_filter: Filter for payload
        :param limit: Maximum number of results
        :param offset: Number of results to skip
        :param sort_by: Sorting options
        :return: SQLAlchemy select statement
        """
        if payload_filter:
            (
                payload_conditions,
                solid_conditions,
            ) = translate_query_to_orm_filters(
                payload_filter, prefix="payload"
            )
        else:
            payload_conditions, solid_conditions = [], []

        payload_conditions.append(cls.db_object_class.original_id.is_(None))

        # First, create a subquery to filter and sort the db_object_class table
        filtered_objects = select(
            cls.db_object_class.object_id,
        ).where(and_(*payload_conditions))

        # Apply sorting to the subquery
        order_criteria = []
        add_columns = []
        if sort_by is not None:
            if sort_by.force_not_payload:
                # Build a raw SQL clause for ordering by a JSON field.
                # This assumes you are using PostgreSQL.
                order_clause = sort_by.field
                add_columns.append(text(sort_by.field))
                if sort_by.order == "asc":
                    order_clause += " ASC"
                else:
                    order_clause += " DESC"
                order_criteria.append(text(order_clause))

            else:
                # Build a raw SQL clause for ordering by a JSON field.
                # This assumes you are using PostgreSQL.
                order_clause = f"payload->'{sort_by.field}'"
                if sort_by.order == "asc":
                    order_clause += " ASC"
                else:
                    order_clause += " DESC"
                order_criteria.append(text(order_clause))

            filtered_objects = filtered_objects.order_by(*order_criteria)

        filtered_objects = filtered_objects.limit(limit)
        if offset is not None:
            filtered_objects = filtered_objects.offset(offset)

        filtered_objects = filtered_objects.alias("filtered_objects")

        # Final query to join with limited results and count parts
        main_query = (
            select(
                filtered_objects.c.object_id,
                func.count(cls.part_id).label("parts_found"),
                cls.db_object_class.payload,
                cls.db_object_class.storage_meta,
                cls.db_object_class.original_id,
                cls.db_object_class.user_id,
                cls.db_object_class.session_id,
                *add_columns,
            )
            .select_from(filtered_objects)
            .join(
                cls,
                filtered_objects.c.object_id == cls.object_id,
                isouter=False,
            )
            .join(
                cls.db_object_class,
                filtered_objects.c.object_id == cls.db_object_class.object_id,
                isouter=False,
            )
            .group_by(
                filtered_objects.c.object_id,
                cls.db_object_class.payload,
                cls.db_object_class.storage_meta,
                cls.db_object_class.original_id,
                cls.db_object_class.user_id,
                cls.db_object_class.session_id,
                *add_columns,
            )
        )
        if sort_by is not None:
            main_query = main_query.order_by(*order_criteria)

        return main_query

    @classmethod
    def insert_parts_statement(cls, db_parts: List["DbObjectPart"]):
        """
        Generate a SQL statement for inserting object parts.

        :param db_parts: List of DbObjectPart instances
        :return: SQLAlchemy insert statement
        """
        db_dicts = cls.db_parts_to_dicts(db_parts, with_metadata=False)
        return insert(cls).values(db_dicts)

    @classmethod
    def upsert_parts_statement(cls, db_parts: List["DbObjectPart"]):
        """
        Generate a SQL statement for upserting object parts.

        :param db_parts: List of DbObjectPart instances
        :return: SQLAlchemy upsert statement
        """
        db_dicts = cls.db_parts_to_dicts(db_parts, with_metadata=False)
        insert_st = pg_insert(cls).values(db_dicts)
        update_dict = {
            "vector": insert_st.excluded.vector,
        }
        return insert_st.on_conflict_do_update(
            index_elements=[cls.part_id], set_=update_dict
        )

    @classmethod
    def delete_statement(cls, object_ids: List[str]):
        """
        Generate a SQL statement for deleting object parts.

        :param object_ids: List of object IDs
        :return: SQLAlchemy delete statement
        """
        return delete(cls).where(cls.object_id.in_(object_ids))

    @classmethod
    def db_part_to_dict(
        cls, db_part: "DbObjectPart", with_metadata: bool = True
    ) -> Dict[str, Any]:
        """
        Convert a DbObjectPart instance to a dictionary.

        :param db_part: DbObjectPart instance
        :param with_metadata: Include metadata in the dictionary
        :return: Dictionary representation of the part
        """
        if with_metadata:
            return {
                "part_id": db_part.part_id,
                "object_id": db_part.object_id,
                "vector": db_part.vector,
                "payload": db_part.object.payload,
                "storage_meta": db_part.object.storage_meta,
                "original_id": db_part.object.original_id,
                "user_id": db_part.object.user_id,
            }
        else:
            return {
                "part_id": db_part.part_id,
                "object_id": db_part.object_id,
                "vector": db_part.vector,
                "user_id": db_part.object.user_id,
            }

    @classmethod
    def db_parts_to_dicts(
        cls, db_parts: List["DbObjectPart"], with_metadata: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Convert a list of DbObjectPart instances to a list of dictionaries.

        :param db_parts: List of DbObjectPart instances
        :param with_metadata: Include metadata in the dictionaries
        :return: List of dictionaries
        """
        return [cls.db_part_to_dict(part, with_metadata) for part in db_parts]

    @classmethod
    def objects_from_db(cls, rows) -> List[Object]:
        """
        Convert database rows to Object instances.

        :param rows: Database result rows
        :return: List of Object instances
        """
        objects_by_id: Dict[str, Object] = {}
        for row in rows:
            obj = objects_by_id.setdefault(
                row.object_id,
                Object(
                    object_id=row.object_id,
                    parts=[],
                    payload=row.payload,
                    storage_meta=row.storage_meta,
                    user_id=row.user_id,
                    original_id=row.original_id,
                ),
            )
            obj.parts.append(
                ObjectPart(
                    part_id=row.part_id,
                    vector=row.vector,
                    is_average=row.is_average,
                )
            )
        return list(objects_by_id.values())

    @classmethod
    def objects_with_distance_from_db(cls, rows) -> List[ObjectWithDistance]:
        """
        Convert database rows to ObjectWithDistance instances.

        :param rows: Database result rows
        :return: List of ObjectWithDistance instances
        """
        objects_by_id: Dict[str, Object] = {}

        for row in rows:
            obj = objects_by_id.setdefault(
                row.object_id,
                ObjectWithDistance(
                    object_id=row.object_id,
                    parts=[],
                    payload=row.payload,
                    storage_meta=row.storage_meta,
                    user_id=row.user_id,
                    original_id=row.original_id,
                    distance=row.distance,
                ),
            )

            if hasattr(row, "vectors"):
                for part_id, vector in zip(
                    row.part_ids, convert_vectors(row.vectors).tolist()
                ):
                    obj.parts.append(
                        ObjectPart(
                            part_id=part_id,
                            vector=vector,
                        )
                    )

            else:
                for part_id in row.part_ids:
                    obj.parts.append(
                        ObjectPart(
                            part_id=part_id,
                            vector=None,
                        )
                    )

        return list(objects_by_id.values())

    @classmethod
    def get_id(cls, row, keep_originals: bool = True):
        """
        Get the ID from a database row, respecting the original ID if present.

        :param row: Database row
        :param keep_originals: If True, return original_id if present
        :return: Object ID
        """
        if keep_originals and row.original_id is not None:
            return row.original_id

        return row.object_id

    @classmethod
    def similar_objects_from_db(
        cls, rows, keep_originals: bool = True
    ) -> List[SimilarObject]:
        """
        Convert database rows to SimilarObject instances.

        :param rows: Database result rows
        :param keep_originals: If True, use original IDs when present
        :return: List of SimilarObject instances
        """
        return [
            SimilarObject(
                object_id=cls.get_id(row, keep_originals),
                distance=row.distance,
                parts_found=row.parts_found
                if hasattr(row, "parts_found")
                else 1,
                payload=row.payload,
                storage_meta=row.storage_meta,
            )
            for row in rows
        ]

    @classmethod
    def found_objects_from_db(
        cls, rows, keep_originals: bool = True
    ) -> List[FoundObject]:
        """
        Convert database rows to FoundObject instances.

        :param rows: Database result rows
        :param keep_originals: If True, use original IDs when present
        :return: List of FoundObject instances
        """
        return [
            FoundObject(
                object_id=cls.get_id(row, keep_originals),
                parts_found=row.parts_found,
                payload=row.payload,
                storage_meta=row.storage_meta,
            )
            for row in rows
        ]

    @classmethod
    def objects_to_db(cls, objects: List[Object]) -> List["DbObjectPart"]:
        """
        Convert Object instances to DbObjectPart instances.

        :param objects: List of Object instances
        :return: List of DbObjectPart instances
        """
        db_objects = []
        for obj in objects:
            db_object = cls.db_object_class(
                object_id=obj.object_id,
                payload=obj.payload,
                storage_meta=obj.storage_meta,
            )
            db_objects.append(db_object)
            for i, part in enumerate(obj.parts):
                part_id = part.part_id or f"{obj.object_id}_{i}"
                try:
                    cls.validate_dimensions(part.vector, cls.search_index)
                except DimensionsMismatch as err:
                    raise RuntimeError(
                        f"Failed to load object part {part_id}"
                    ) from err
                db_objects.append(
                    cls(
                        object_id=obj.object_id,
                        part_id=part_id,
                        vector=part.vector,
                        object=db_object,
                    )
                )
        return db_objects

    @classmethod
    def find_by_session_id_statement(cls, session_id: str):
        """
        Create a SQLAlchemy statement to retrieve objects and their parts filtered by session ID.

        This query starts with the DbObject table and left joins with the DbObjectPart table
        to return all objects where the session_id matches the provided value,
        along with any parts those objects might have.

        :param session_id: The session ID to filter objects by
        :return: SQLAlchemy select statement
        """
        return (
            select(
                cls.db_object_class.object_id,
                cls.part_id,
                cls.vector,
                cls.is_average,
                cls.db_object_class.payload,
                cls.db_object_class.storage_meta,
                cls.db_object_class.original_id,
                cls.db_object_class.user_id,
                cls.db_object_class.session_id,
            )
            .select_from(cls.db_object_class)
            .join(
                cls,
                cls.object_id == cls.db_object_class.object_id,
                isouter=True,  # This makes it a LEFT OUTER JOIN
            )
            .where(cls.db_object_class.session_id == session_id)
        )


def get_dbo_table_name(collection_info: CollectionInfo) -> Dict[str, str]:
    """
    Generate database table and index names for a collection.

    :param collection_info: CollectionInfo object
    :return: Dictionary of table and index names
    """
    return {
        "dbo_collection": f"dbo_{collection_info.collection_id}",
        "dbop_collection": f"dbop_{collection_info.collection_id}",
        "dbo_relation": f"DbObject_{collection_info.collection_id}",
        "dbop_relation": f"DbObjectPart_{collection_info.collection_id}",
        "dbo_payload_index": f"ix_dbo_{collection_info.collection_id}_p",
        "dbo_session_id_index": f"ix_{collection_info.collection_id}_sid",
        "dbo_original_id_index": f"ix_{collection_info.collection_id}_oid",
        "dbo_user_id_index": f"ix_{collection_info.collection_id}_uid",
    }


def make_db_model(
    collection_info: CollectionInfo,
) -> Tuple[Type[DbObjectBase], Type[DbObjectPartBase]]:
    """
    Create database model classes for a collection.

    Dynamically creates DbObject and DbObjectPart classes tailored to the
    specific collection, with appropriate table names and relationships.

    :param collection_info: CollectionInfo object
    :return: Tuple of (DbObject class, DbObjectPart class)
    """
    search_index = collection_info.embedding_model
    collection_id = collection_info.collection_id

    _names = get_dbo_table_name(collection_info)

    class DbObject(DbObjectBase, DbObjectImpl):
        __tablename__ = _names["dbo_collection"]

        __table_args__ = (
            Index(
                _names["dbo_payload_index"],
                "payload",
                postgresql_using="gin",
            ),
            Index(
                _names["dbo_session_id_index"],
                "session_id",
            ),
            Index(
                _names["dbo_original_id_index"],
                "original_id",
            ),
            Index(
                _names["dbo_user_id_index"],
                "user_id",
            ),
            {"extend_existing": True},
        )

    class DbObjectPart(DbObjectPartBase, DbObjectPartImpl):
        __tablename__ = _names["dbop_collection"]
        object_id = mapped_column(
            String(128),
            ForeignKey(f"{_names['dbo_collection']}.object_id"),
            index=True,
        )
        vector = mapped_column(Vector(search_index.dimensions))

        __table_args__ = {"extend_existing": True}

        @classmethod
        def hnsw_index(cls):
            if cls.search_index.metric_type is MetricType.COSINE:
                index_type = "vector_cosine_ops"
            elif cls.search_index.metric_type is MetricType.DOT:
                index_type = "vector_ip_ops"
            elif cls.search_index.metric_type is MetricType.EUCLID:
                index_type = "vector_l2_ops"
            else:
                raise RuntimeError(
                    f"Unknown metric type: {cls.search_index.metric_type}"
                )

            return Index(
                f"{collection_info.collection_id}_index",
                cls.vector,
                postgresql_using="hnsw",
                postgresql_with={
                    "m": cls.search_index.hnsw.m,
                    "ef_construction": cls.search_index.hnsw.ef_construction,
                },
                postgresql_ops={"vector": index_type},
            )

    DbObject.__name__ = f"DbObject_{collection_id}"
    DbObjectPart.__name__ = f"DbObjectPart_{collection_id}"

    DbObject.parts = relationship(
        DbObjectPart, back_populates="object", cascade="all, delete-orphan"
    )
    DbObjectPart.object = relationship(DbObject, back_populates="parts")

    DbObjectPart.initialize(search_index, DbObject)

    return DbObject, DbObjectPart
