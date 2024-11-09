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
    def get_total_statement(cls):
        return select(func.count(cls.object_id))

    @classmethod
    def get_objects_common_data_batch_statement(
        cls, limit: int, offset: Optional[int] = None
    ):
        return (
            select(cls.object_id, cls.payload, cls.storage_meta)
            .limit(limit)
            .offset(offset)
        )

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
    @classmethod
    def create_table(cls, pg_database: sqlalchemy.Engine):
        cls.__table__.create(pg_database, checkfirst=True)

    @classmethod
    def validate_dimensions(cls, vector: List[float], search_index):
        dim = len(vector)
        if dim != search_index.dimensions:
            raise DimensionsMismatch(
                f"Dimensions mismatch: input vector({dim}), expected vector({search_index.dimensions})"
            )

    @classmethod
    def distance_expression(cls, vector: List[float], search_index):
        cls.validate_dimensions(vector, search_index)
        if search_index.metric_type is MetricType.COSINE:
            return cls.vector.cosine_distance(vector)
        if search_index.metric_type is MetricType.DOT:
            return cls.vector.max_inner_product(vector)
        if search_index.metric_type is MetricType.EUCLID:
            return cls.vector.l2_distance(vector)
        raise RuntimeError(
            f"unknown metric type: {search_index.metric_type.value}"
        )

    @classmethod
    def aggregated_distance_expression(cls, vector: List[float], search_index):
        dst = cls.distance_expression(vector, search_index)
        if search_index.metric_aggregation_type is MetricAggregationType.AVG:
            return func.avg(dst)
        elif search_index.metric_aggregation_type is MetricAggregationType.MIN:
            return func.min(dst)
        raise RuntimeError(
            f"unknown metric aggregation type: {search_index.metric_aggregation_type.value}"
        )

    @classmethod
    def find_by_id_statement(cls, object_ids: List[float], DbObjectClass):
        return (
            select(
                DbObjectClass.object_id,
                cls.part_id,
                cls.vector,
                DbObjectClass.payload,
                DbObjectClass.storage_meta,
            )
            .join(DbObjectClass)
            .where(DbObjectClass.object_id.in_(object_ids))
        )

    @classmethod
    def similarity_search_statement(
        cls,
        query_vector: List[float],
        limit: int,
        offset: Optional[int],
        max_distance: Optional[float],
        payload_filter: Optional[PayloadFilter],
        search_index,
        DbObjectClass,
    ):
        adist_expr = cls.aggregated_distance_expression(
            query_vector, search_index
        )
        select_st = (
            select(
                DbObjectClass.object_id,
                adist_expr.label("distance"),
                func.count(cls.part_id).label("parts_found"),
                DbObjectClass.payload,
                DbObjectClass.storage_meta,
            )
            .join(DbObjectClass)
            .group_by(
                DbObjectClass.object_id,
                DbObjectClass.payload,
                DbObjectClass.storage_meta,
            )
            .order_by(asc("distance"))
            .limit(limit)
        )
        conditions = []
        if max_distance is not None:
            dist_expr = cls.distance_expression(query_vector, search_index)
            conditions.append(dist_expr < max_distance)

        if payload_filter:
            payload_conditions = translate_query_to_orm_filters(
                payload_filter, prefix="payload"
            )
            conditions += payload_conditions

        if conditions:
            select_st = select_st.where(and_(*conditions))

        if offset is not None:
            select_st = select_st.offset(offset)

        return select_st

    @classmethod
    def payload_search_statement(
        cls,
        payload_filter: PayloadFilter,
        limit: int,
        offset: Optional[int],
        DbObjectClass,
    ):
        payload_conditions = translate_query_to_orm_filters(
            payload_filter, prefix="payload"
        )

        select_statement = (
            select(
                DbObjectClass.object_id,
                func.count(cls.part_id).label("parts_found"),
                DbObjectClass.payload,
                DbObjectClass.storage_meta,
            )
            .join(DbObjectClass)
            .where(and_(*payload_conditions))
            .group_by(
                DbObjectClass.object_id,
                DbObjectClass.payload,
                DbObjectClass.storage_meta,
            )
            .limit(limit)
        )

        if offset is not None:
            select_statement = select_statement.offset(offset)

        return select_statement

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
    def objects_to_db(
        cls, objects: List[Object], DbObjectClass
    ) -> List["DbObjectPart"]:
        db_objects = []
        for obj in objects:
            db_object = DbObjectClass(
                object_id=obj.object_id,
                payload=obj.payload,
                storage_meta=obj.storage_meta,
            )
            db_objects.append(db_object)
            for i, part in enumerate(obj.parts):
                part_id = part.part_id or f"{obj.object_id}_{i}"
                try:
                    cls.validate_dimensions(part.vector, search_index)
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

    @classmethod
    def similar_objects_from_db(cls, rows) -> List[SimilarObject]:
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

    @classmethod
    def found_objects_from_db(cls, rows) -> List[FoundObject]:
        return [
            FoundObject(
                object_id=row.object_id,
                parts_found=row.parts_found,
                payload=row.payload,
                storage_meta=row.storage_meta,
            )
            for row in rows
        ]


def make_db_model(
    collection_info: CollectionInfo,
) -> Tuple[Type[DbObjectBase], Type[DbObjectPartBase]]:
    search_index = collection_info.search_index_info
    collection_id = collection_info.collection_id

    class DbObject(DbObjectBase, DbObjectImpl):
        __tablename__ = f"dbo_{collection_id}"

        parts = relationship(
            f"DbObjectPart_{collection_id}",
            back_populates="object",
            cascade="all, delete-orphan",
        )

        __table_args__ = (
            Index(
                f"idx_{collection_id}",
                "payload",
                postgresql_using="gin",
            ),
        )

    class DbObjectPart(DbObjectPartBase, DbObjectPartImpl):
        __tablename__ = f"dbop_{collection_id}"
        object_id = mapped_column(
            String(128),
            ForeignKey(f"dbo_{collection_id}.object_id"),
            index=True,
        )
        vector = mapped_column(Vector(search_index.dimensions))
        object = relationship(
            f"DbObject_{collection_id}", back_populates="parts"
        )

        @classmethod
        def hnsw_index(cls):
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
                cls.vector,
                postgresql_using="hnsw",
                postgresql_with={
                    "m": search_index.hnsw.m,
                    "ef_construction": search_index.hnsw.ef_construction,
                },
                postgresql_ops={"vector": index_type},
            )

    DbObject.__name__ = f"DbObject_{collection_id}"
    DbObjectPart.__name__ = f"DbObjectPart_{collection_id}"

    return DbObject, DbObjectPart
