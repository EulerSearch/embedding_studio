import logging
from typing import Any, Dict, List, Optional, Tuple, Type

import sqlalchemy
from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    ForeignKey,
    Index,
    String,
    and_,
    asc,
    delete,
    insert,
    select,
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
from embedding_studio.vectordb.pgvector.query_to_sql import (
    translate_query_to_orm_filters,
)

logger = logging.getLogger(__name__)


class DimensionsMismatch(Exception):
    pass


Base = declarative_base()


class DbObjectBase(Base):
    __abstract__ = True

    @declared_attr
    def object_id(cls):
        return mapped_column(String(128), primary_key=True)

    @declared_attr
    def payload(cls):
        return mapped_column(JSONB)

    @declared_attr
    def storage_meta(cls):
        return mapped_column(JSONB)

    @declared_attr
    def original_id(cls):
        return mapped_column(String(128))

    @declared_attr
    def user_id(cls):
        return mapped_column(String(128))

    @declared_attr
    def session_id(cls):
        return mapped_column(String(128))


class DbObjectPartBase(Base):
    __abstract__ = True

    @declared_attr
    def part_id(cls):
        return mapped_column(String(128), primary_key=True)

    @declared_attr
    def object_id(cls):
        return mapped_column(
            String(128),
            ForeignKey(f"dbo_{cls.__tablename__[5:]}.object_id"),
            index=True,
        )

    @declared_attr
    def vector(cls):
        return mapped_column(Vector)


class DbObjectImpl:
    @classmethod
    def create_table(cls, pg_database: sqlalchemy.Engine):
        cls.__table__.create(pg_database, checkfirst=True)

    @classmethod
    def insert_objects_statement(cls, db_objects: List["DbObject"]):
        db_dicts = cls.db_objects_to_dicts(db_objects)
        return insert(cls).values(db_dicts)

    @classmethod
    def upsert_objects_statement(cls, db_objects: List["DbObject"]):
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
        return {
            "object_id": db_object.object_id,
            "payload": db_object.payload,
            "storage_meta": db_object.storage_meta,
            "original_id": db_object.original_id,
            "user_id": db_object.user_id,
        }

    @classmethod
    def db_objects_to_dicts(
        cls, db_objects: List["DbObject"]
    ) -> List[Dict[str, Any]]:
        return [cls.db_object_to_dict(obj) for obj in db_objects]

    @classmethod
    def delete_statement(cls, object_ids: List[str]):
        return delete(cls).where(cls.object_id.in_(object_ids))

    @classmethod
    def get_total_statement(cls, originals_only: bool = True):
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
        return [
            ObjectСommonData(
                object_id=row.object_id,
                payload=row.payload,
                storage_meta=row.storage_meta,
            )
            for row in rows
        ]


class DbObjectPartImpl:
    search_index = None
    db_object_class = None

    @classmethod
    def initialize(cls, search_index, db_object_class):
        cls.search_index = search_index
        cls.db_object_class = db_object_class

    @classmethod
    def create_table(cls, pg_database: sqlalchemy.Engine):
        cls.__table__.create(pg_database, checkfirst=True)

    @classmethod
    def validate_dimensions(cls, vector: List[float]):
        dim = len(vector)
        if dim != cls.search_index.dimensions:
            raise DimensionsMismatch(
                f"Dimensions mismatch: input vector({dim}), expected vector({cls.search_index.dimensions})"
            )

    @classmethod
    def distance_expression(cls, vector: List[float]):
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
            .where(cls.db_object_class.object_id.in_(object_ids))
        )

    @classmethod
    def find_by_original_id_statement(cls, object_ids: List[float]):
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
        user_id: Optional[str] = None,  # Add user_id as an optional parameter
        with_vectors: bool = False,
    ):
        # TODO: Profile query performance on large collections to measure execution times.
        # TODO: Test scalability by simulating high user load and frequent vector additions.
        # TODO: Ensure indexing strategy is optimized for frequently used fields (e.g., object_id, user_id).
        # TODO: Plan for database growth by implementing cleanup, archiving, or sharding strategies.
        # TODO: Explore ways to simplify this query, potentially by breaking it into smaller components.
        # TODO: Set up monitoring and alerts for performance degradation and database size issues.
        # TODO: Document the query logic and its potential scalability impact for maintainability.
        # TODO: Investigate alternative search backends for handling large-scale vector operations.

        adist_expr = cls.aggregated_distance_expression(query_vector)

        # Subquery to identify modified copies
        subquery = (
            select(cls.db_object_class.object_id)
            .where(
                cls.db_object_class.original_id.isnot(None)
            )  # Find modified copies
            .distinct()
            .subquery()
        )

        selection_part = None
        group_by = []
        if with_vectors:
            selection_part = select(
                cls.db_object_class.object_id,
                adist_expr.label("distance"),
                func.count(cls.part_id).label("parts_found"),
                cls.db_object_class.payload,
                cls.db_object_class.storage_meta,
                cls.db_object_class.original_id,
                cls.db_object_class.user_id,
                cls.db_object_class.session_id,
                cls.vector,
                cls.part_id,
            )
            group_by = [
                cls.db_object_class.object_id,
                cls.db_object_class.payload,
                cls.db_object_class.storage_meta,
                cls.db_object_class.original_id,
                cls.db_object_class.user_id,
                cls.db_object_class.session_id,
                cls.vector,
                cls.part_id,
            ]
        else:
            selection_part = select(
                cls.db_object_class.object_id,
                adist_expr.label("distance"),
                func.count(cls.part_id).label("parts_found"),
                cls.db_object_class.payload,
                cls.db_object_class.storage_meta,
                cls.db_object_class.original_id,
                cls.db_object_class.user_id,
                cls.db_object_class.session_id,
            )
            group_by = [
                cls.db_object_class.object_id,
                cls.db_object_class.payload,
                cls.db_object_class.storage_meta,
                cls.db_object_class.original_id,
                cls.db_object_class.user_id,
                cls.db_object_class.session_id,
            ]

        # Base query excluding original objects that have modified copies
        select_st = (
            selection_part.join(cls.db_object_class)
            .where(
                ~cls.db_object_class.object_id.in_(subquery)
            )  # Exclude originals with copies
            .group_by(*group_by)
            .limit(limit)
        )

        # Add filtering conditions
        conditions = []
        if max_distance is not None:
            dist_expr = cls.distance_expression(query_vector)
            conditions.append(dist_expr < max_distance)

            if cls.search_index.metric_type in [
                MetricType.COSINE,
                MetricType.DOT,
            ]:
                # For similarity-based metrics, threshold should be greater than or equal
                conditions.append(dist_expr >= max_distance)

            elif cls.search_index.metric_type == MetricType.EUCLID:
                # For distance-based metrics, threshold should be less than or equal
                conditions.append(dist_expr <= max_distance)

        if payload_filter:
            payload_conditions = translate_query_to_orm_filters(
                payload_filter, prefix="payload"
            )
            conditions += payload_conditions

        if conditions:
            select_st = select_st.where(and_(*conditions))

        if offset is not None:
            select_st = select_st.offset(offset)

        # Prioritize rows with the specified user_id
        if user_id:
            select_st = select_st.order_by(
                sqlalchemy.case(
                    [(cls.db_object_class.user_id == user_id, 0)], else_=1
                ),
                asc("distance"),
            )
        else:
            select_st = select_st.order_by(asc("distance"))

        return select_st

    @classmethod
    def payload_search_statement(
        cls,
        payload_filter: PayloadFilter,
        limit: int,
        offset: Optional[int] = None,
    ):
        # Translate payload filters to ORM conditions
        payload_conditions = translate_query_to_orm_filters(
            payload_filter, prefix="payload"
        )

        # Add condition to include only original objects (original_id is NULL)
        payload_conditions.append(cls.db_object_class.original_id.is_(None))

        # Construct the select statement
        select_statement = (
            select(
                cls.db_object_class.object_id,
                func.count(cls.part_id).label("parts_found"),
                cls.db_object_class.payload,
                cls.db_object_class.storage_meta,
                cls.db_object_class.original_id,
                cls.db_object_class.user_id,
                cls.db_object_class.session_id,
            )
            .join(cls.db_object_class)
            .where(and_(*payload_conditions))  # Apply payload filters
            .group_by(
                cls.db_object_class.object_id,
                cls.db_object_class.payload,
                cls.db_object_class.storage_meta,
                cls.db_object_class.original_id,
                cls.db_object_class.user_id,
                cls.db_object_class.session_id,
            )
            .limit(limit)
        )

        # Apply offset if specified
        if offset is not None:
            select_statement = select_statement.offset(offset)

        return select_statement

    @classmethod
    def find_by_session_id_statement(cls, session_id: str):
        return (
            select(
                cls.db_object_class.object_id,
                cls.db_object_class.payload,
                cls.db_object_class.storage_meta,
                cls.db_object_class.original_id,
                cls.db_object_class.user_id,
                cls.db_object_class.session_id,
                cls.vector,
                cls.part_id,
            )
            .join(
                cls.db_object_class,
                cls.object_id == cls.db_object_class.object_id,
            )
            .where(cls.db_object_class.session_id.astext == session_id)
        )

    @classmethod
    def insert_parts_statement(cls, db_parts: List["DbObjectPart"]):
        db_dicts = cls.db_parts_to_dicts(db_parts, with_metadata=False)
        return insert(cls).values(db_dicts)

    @classmethod
    def upsert_parts_statement(cls, db_parts: List["DbObjectPart"]):
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
        return delete(cls).where(cls.object_id.in_(object_ids))

    @classmethod
    def db_part_to_dict(
        cls, db_part: "DbObjectPart", with_metadata: bool = True
    ) -> Dict[str, Any]:
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
            }

    @classmethod
    def db_parts_to_dicts(
        cls, db_parts: List["DbObjectPart"], with_metadata: bool = True
    ) -> List[Dict[str, Any]]:
        return [cls.db_part_to_dict(part, with_metadata) for part in db_parts]

    @classmethod
    def objects_from_db(cls, rows) -> List[Object]:
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
                )
            )
        return list(objects_by_id.values())

    @classmethod
    def objects_with_distance_from_db(cls, rows) -> List[ObjectWithDistance]:
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
            obj.parts.append(
                ObjectPart(
                    part_id=row.part_id,
                    vector=row.vector,
                )
            )
        return list(objects_by_id.values())

    @classmethod
    def get_id(cls, row, keep_originals: bool = True):
        if keep_originals and row.original_id is not None:
            return row.original_id

        return row.object_id

    @classmethod
    def similar_objects_from_db(
        cls, rows, keep_originals: bool = True
    ) -> List[SimilarObject]:
        return [
            SimilarObject(
                object_id=cls.get_id(row, keep_originals),
                distance=row.distance,
                parts_found=row.parts_found,
                payload=row.payload,
                storage_meta=row.storage_meta,
            )
            for row in rows
        ]

    @classmethod
    def found_objects_from_db(
        cls, rows, keep_originals: bool = True
    ) -> List[FoundObject]:
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


def get_dbo_table_name(collection_info: CollectionInfo) -> Dict[str, str]:
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
