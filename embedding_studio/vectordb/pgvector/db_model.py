import logging
from typing import Any, Dict, List, Optional, Type

from pgvector.sqlalchemy import Vector
from sqlalchemy import Index, String, asc, delete, insert, select
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from embedding_studio.models.embeddings.collections import CollectionInfo
from embedding_studio.models.embeddings.models import (
    MetricAggregationType,
    MetricType,
)
from embedding_studio.models.embeddings.objects import (
    Object,
    ObjectPart,
    SimilarObject,
)

logger = logging.getLogger(__name__)


class DimensionsMismatch(Exception):
    pass


Base = declarative_base()


class DbObjectPartBase(Base):
    __abstract__ = True
    part_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    object_id: Mapped[str] = mapped_column(String(128), index=True)


DB_MODELS: Dict[str, Type[DbObjectPartBase]] = {}


def _make_db_model(collection_info: CollectionInfo):
    collection_info.embedding_model
    search_index = collection_info.search_index_info

    class DbObjectPart(DbObjectPartBase):
        __tablename__ = collection_info.collection_id
        vector = mapped_column(Vector(search_index.dimensions))

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
            return select(
                DbObjectPart.object_id,
                DbObjectPart.part_id,
                DbObjectPart.vector,
            ).where(DbObjectPart.object_id.in_(object_ids))

        @staticmethod
        def similarity_search_statement(
            query_vector: List[float],
            limit: int,
            offset: Optional[int] = None,
            max_distance: Optional[float] = None,
        ):
            adist_expr = DbObjectPart.aggregated_distance_expression(
                query_vector
            )
            select_st = (
                select(
                    DbObjectPart.object_id,
                    adist_expr.label("distance"),
                    func.count(DbObjectPart.part_id).label("parts_found"),
                )
                .group_by(DbObjectPart.object_id)
                .order_by(asc("distance"))
                .limit(limit)
            )
            if max_distance:
                dist_expr = DbObjectPart.distance_expression(query_vector)
                select_st = select_st.where(dist_expr < max_distance)
            if offset:
                select_st = select_st.offset(offset)
            return select_st

        @staticmethod
        def insert_statement(db_object_parts: List["DbObjectPart"]):
            db_dicts = DbObjectPart.db_objects_to_dicts(db_object_parts)
            return insert(DbObjectPart).values(db_dicts)

        @staticmethod
        def upsert_statement(db_object_parts: List["DbObjectPart"]):
            insert_st = DbObjectPart.insert_statement(db_object_parts)
            return insert_st.on_conflict_do_update(
                index_elements=[DbObjectPart.part_id],
                set_=dict(vector=insert_st.excluded.vector),
            )

        @staticmethod
        def delete_statement(object_ids: List[str]):
            return delete(DbObjectPart).where(
                DbObjectPart.object_id.in_(object_ids)
            )

        @staticmethod
        def objects_to_db(objects: List[Object]) -> List["DbObjectPart"]:
            result: List[DbObjectPart] = []
            for obj in objects:
                for i, part in enumerate(obj.parts):
                    part_id = part.part_id or f"{obj.object_id}_{i}"
                    try:
                        DbObjectPart.validate_dimensions(part.vector)
                    except DimensionsMismatch as err:
                        raise RuntimeError(
                            f"Failed to load object part {part_id}"
                        ) from err
                    result.append(
                        DbObjectPart(
                            object_id=obj.object_id,
                            part_id=part_id,
                            vector=part.vector,
                        )
                    )
            return result

        @staticmethod
        def db_object_to_dict(db_object: "DbObjectPart") -> Dict[str, Any]:
            result = db_object.__dict__
            result.pop("_sa_instance_state")
            return result

        @staticmethod
        def db_objects_to_dicts(
            db_objects: List["DbObjectPart"],
        ) -> List[Dict[str, Any]]:
            return [DbObjectPart.db_object_to_dict(obj) for obj in db_objects]

        @staticmethod
        def objects_from_db(rows) -> List[Object]:
            objects_by_id: Dict[str, Object] = {}
            for row in rows:
                obj = objects_by_id.setdefault(
                    row.object_id, Object(object_id=row.object_id, parts=[])
                )
                obj.parts.append(
                    ObjectPart(part_id=row.part_id, vector=row.vector)
                )
            return list(objects_by_id.values())

        @staticmethod
        def similar_objects_from_db(rows) -> List[SimilarObject]:
            return [
                SimilarObject(
                    object_id=row.object_id,
                    distance=row.distance,
                    parts_found=row.parts_found,
                )
                for row in rows
            ]

    return DbObjectPart


def make_db_model(collection_info: CollectionInfo):
    col_id = collection_info.collection_id
    if col_id in DB_MODELS:
        return DB_MODELS[col_id]
    model = _make_db_model(collection_info)
    DB_MODELS[col_id] = model
    return model
