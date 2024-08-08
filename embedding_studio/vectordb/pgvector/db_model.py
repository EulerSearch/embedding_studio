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
        return mapped_column(JSONB, index=True)

    @declared_attr
    def storage_meta(cls):
        return mapped_column(JSONB)


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


DB_MODELS: Dict[str, Tuple[Type[DbObjectBase], Type[DbObjectPartBase]]] = {}


def _make_db_model(collection_info: CollectionInfo):
    search_index = collection_info.search_index_info

    class DbObject(DbObjectBase):
        __tablename__ = f"dbo_{collection_info.collection_id}"

        parts = relationship(
            "DbObjectPart",
            back_populates="object",
            cascade="all, delete-orphan",
        )

        __table_args__ = (
            Index(
                f"idx_{collection_info.collection_id}",
                "payload",
                postgresql_using="gin",
            ),
        )

        @classmethod
        def create_table(cls, pg_database: sqlalchemy.Engine):
            cls.__table__.create(pg_database, checkfirst=True)

        @staticmethod
        def insert_objects_statement(db_objects: List["DbObject"]):
            db_dicts = DbObject.db_objects_to_dicts(db_objects)
            return insert(DbObject).values(db_dicts)

        @staticmethod
        def upsert_objects_statement(db_objects: List["DbObject"]):
            db_dicts = DbObject.db_objects_to_dicts(db_objects)
            insert_st = pg_insert(DbObject).values(db_dicts)
            update_dict = {
                "payload": insert_st.excluded.payload,
                "storage_meta": insert_st.excluded.storage_meta,
            }
            return insert_st.on_conflict_do_update(
                index_elements=[DbObject.object_id],
                set_=update_dict,
            )

        @staticmethod
        def db_object_to_dict(db_object: "DbObject") -> Dict[str, Any]:
            return {
                "object_id": db_object.object_id,
                "payload": db_object.payload,
                "storage_meta": db_object.storage_meta,
            }

        @staticmethod
        def db_objects_to_dicts(
            db_objects: List["DbObject"],
        ) -> List[Dict[str, Any]]:
            return [DbObject.db_object_to_dict(obj) for obj in db_objects]

        @staticmethod
        def delete_statement(object_ids: List[str]):
            return delete(DbObject).where(DbObject.object_id.in_(object_ids))

    class DbObjectPart(DbObjectPartBase):
        __tablename__ = f"dbop_{collection_info.collection_id}"
        object_id = mapped_column(
            String(128),
            ForeignKey(f"dbo_{collection_info.collection_id}.object_id"),
            index=True,
        )
        vector = mapped_column(Vector(search_index.dimensions))
        object = relationship("DbObject", back_populates="parts")

        @classmethod
        def create_table(cls, pg_database: sqlalchemy.Engine):
            cls.__table__.create(pg_database, checkfirst=True)

        @staticmethod
        def hnsw_index():
            if search_index.metric_type is MetricType.COSINE:
                index_type = "vector_cosine_ops"
            elif search_index.metric_type is MetricType.DOT:
                index_type = "vector_ip_ops"
            elif search_index.metric_type is MetricType.EUCLID:
                index_type = "vector_l2_ops"
            else:
                raise RuntimeError(
                    f"Unknown metric type: {search_index.metric_type}"
                )

            return Index(
                f"{collection_info.collection_id}_index",
                DbObjectPart.vector,
                postgresql_using="hnsw",
                postgresql_with={
                    "m": search_index.hnsw.m,
                    "ef_construction": search_index.hnsw.ef_construction,
                },
                postgresql_ops={"vector": index_type},
            )

        @staticmethod
        def validate_dimensions(vector: List[float]):
            dim = len(vector)
            if dim != search_index.dimensions:
                raise DimensionsMismatch(
                    f"Dimensions mismatch: "
                    f"input vector({dim}), expected vector({search_index.dimensions})"
                )

        @staticmethod
        def distance_expression(vector: List[float]):
            DbObjectPart.validate_dimensions(vector)
            if search_index.metric_type is MetricType.COSINE:
                return DbObjectPart.vector.cosine_distance(vector)
            if search_index.metric_type is MetricType.DOT:
                return DbObjectPart.vector.max_inner_product(vector)
            if search_index.metric_type is MetricType.EUCLID:
                return DbObjectPart.vector.l2_distance(vector)
            raise RuntimeError(
                f"unknown metric type: {search_index.metric_type.value}"
            )

        @staticmethod
        def aggregated_distance_expression(vector: List[float]):
            dst = DbObjectPart.distance_expression(vector)
            if (
                search_index.metric_aggregation_type
                is MetricAggregationType.AVG
            ):
                return func.avg(dst)
            elif (
                search_index.metric_aggregation_type
                is MetricAggregationType.MIN
            ):
                return func.min(dst)
            raise RuntimeError(
                f"unknown metric aggregation type: {search_index.metric_aggregation_type.value}"
            )

        @staticmethod
        def find_by_id_statement(object_ids: List[float]):
            return (
                select(
                    DbObject.object_id,
                    DbObjectPart.part_id,
                    DbObjectPart.vector,
                    DbObject.payload,
                    DbObject.storage_meta,
                )
                .join(DbObject)
                .where(DbObject.object_id.in_(object_ids))
            )

        @staticmethod
        def similarity_search_statement(
            query_vector: List[float],
            limit: int,
            offset: Optional[int] = None,
            max_distance: Optional[float] = None,
            payload_filter: Optional[PayloadFilter] = None,
        ):
            adist_expr = DbObjectPart.aggregated_distance_expression(
                query_vector
            )
            select_st = (
                select(
                    DbObject.object_id,
                    adist_expr.label("distance"),
                    func.count(DbObjectPart.part_id).label("parts_found"),
                    DbObject.payload,
                    DbObject.storage_meta,
                )
                .join(DbObject)
                .group_by(
                    DbObject.object_id, DbObject.payload, DbObject.storage_meta
                )
                .order_by(asc("distance"))
                .limit(limit)
            )
            conditions = []
            if max_distance is not None:
                dist_expr = DbObjectPart.distance_expression(query_vector)
                conditions.append(dist_expr < max_distance)

            if payload_filter:
                payload_conditions = translate_query_to_orm_filters(
                    payload_filter, prefix="payload"
                )
                conditions += payload_conditions

            # Apply conditions to the select statement
            if conditions:
                select_st = select_st.where(and_(*conditions))

            # Adding offset if specified
            if offset is not None:
                select_st = select_st.offset(offset)

            return select_st

        @staticmethod
        def payload_search_statement(
            payload_filter: PayloadFilter,
            limit: int,
            offset: Optional[int] = None,
        ):
            # Translate the payload query into a SQL condition and convert it to a SQLAlchemy text clause
            payload_conditions = translate_query_to_orm_filters(
                payload_filter, prefix="payload"
            )

            # Create the base select statement
            select_statement = (
                select(
                    DbObject.object_id,
                    func.count(DbObjectPart.part_id).label("parts_found"),
                    DbObject.payload,
                    DbObject.storage_meta,
                )
                .join(DbObject)
                .where(and_(*payload_conditions))
                .group_by(
                    DbObject.object_id, DbObject.payload, DbObject.storage_meta
                )
                .limit(limit)
            )

            if offset is not None:
                select_statement = select_statement.offset(offset)

            return select_statement

        @staticmethod
        def insert_parts_statement(db_parts: List["DbObjectPart"]):
            db_dicts = DbObjectPart.db_parts_to_dicts(
                db_parts, with_metadata=False
            )
            return insert(DbObjectPart).values(db_dicts)

        @staticmethod
        def upsert_parts_statement(db_parts: List["DbObjectPart"]):
            db_dicts = DbObjectPart.db_parts_to_dicts(
                db_parts, with_metadata=False
            )

            insert_st = pg_insert(DbObjectPart).values(db_dicts)
            update_dict = {
                "vector": insert_st.excluded.vector,
            }
            return insert_st.on_conflict_do_update(
                index_elements=[DbObjectPart.part_id],
                set_=update_dict,
            )

        @staticmethod
        def delete_statement(object_ids: List[str]):
            return delete(DbObjectPart).where(
                DbObjectPart.object_id.in_(object_ids)
            )

        @staticmethod
        def objects_to_db(objects: List[Object]) -> List["DbObjectPart"]:
            db_objects = []
            for obj in objects:
                db_object = DbObject(
                    object_id=obj.object_id,
                    payload=obj.payload,
                    storage_meta=obj.storage_meta,
                )
                db_objects.append(db_object)
                for i, part in enumerate(obj.parts):
                    part_id = part.part_id or f"{obj.object_id}_{i}"
                    try:
                        DbObjectPart.validate_dimensions(part.vector)
                    except DimensionsMismatch as err:
                        raise RuntimeError(
                            f"Failed to load object part {part_id}"
                        ) from err
                    db_objects.append(
                        DbObjectPart(
                            object_id=obj.object_id,
                            part_id=part_id,
                            vector=part.vector,
                            object=db_object,
                        )
                    )
            return db_objects

        @staticmethod
        def validate_dimensions(vector: List[float]):
            dim = len(vector)
            if dim != search_index.dimensions:
                raise DimensionsMismatch(
                    f"Dimensions mismatch: "
                    f"input vector({dim}), expected vector({search_index.dimensions})"
                )

        @staticmethod
        def db_part_to_dict(
            db_part: "DbObjectPart", with_metadata: bool = True
        ) -> Dict[str, Any]:
            if with_metadata:
                return {
                    "part_id": db_part.part_id,
                    "object_id": db_part.object_id,
                    "vector": db_part.vector,
                    "payload": db_part.object.payload,
                    "storage_meta": db_part.object.storage_meta,
                }
            else:
                return {
                    "part_id": db_part.part_id,
                    "object_id": db_part.object_id,
                    "vector": db_part.vector,
                }

        @staticmethod
        def db_parts_to_dicts(
            db_parts: List["DbObjectPart"], with_metadata: bool = True
        ) -> List[Dict[str, Any]]:
            return [
                DbObjectPart.db_part_to_dict(part, with_metadata)
                for part in db_parts
            ]

        @staticmethod
        def objects_from_db(rows) -> List[Object]:
            objects_by_id: Dict[str, Object] = {}
            for row in rows:
                obj = objects_by_id.setdefault(
                    row.object_id, Object(object_id=row.object_id, parts=[])
                )
                obj.parts.append(
                    ObjectPart(
                        part_id=row.part_id,
                        vector=row.vector,
                        payload=obj.payload,
                    )
                )
            return list(objects_by_id.values())

        @staticmethod
        def similar_objects_from_db(rows) -> List[SimilarObject]:
            return [
                SimilarObject(
                    object_id=row.object_id,
                    distance=row.distance,
                    parts_found=row.parts_found,
                    payload=row.payload,
                    storage_meta=row.storage_meta,
                )
                for row in rows
            ]

        @staticmethod
        def found_objects_from_db(rows) -> List[FoundObject]:
            return [
                FoundObject(
                    object_id=row.object_id,
                    parts_found=row.parts_found,
                    payload=row.payload,
                    storage_meta=row.storage_meta,
                )
                for row in rows
            ]

    return DbObject, DbObjectPart


def make_db_model(collection_info: CollectionInfo):
    col_id = collection_info.collection_id
    if col_id in DB_MODELS:
        return DB_MODELS[col_id]
    object_model, object_part_model = _make_db_model(collection_info)
    DB_MODELS[col_id] = (object_model, object_part_model)
    return object_model, object_part_model
