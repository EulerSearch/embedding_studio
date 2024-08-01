import logging
from typing import List, Optional

import sqlalchemy

from embedding_studio.models.embeddings.collections import CollectionStateInfo
from embedding_studio.models.embeddings.objects import Object, SearchResults
from embedding_studio.models.payload.models import PayloadFilter
from embedding_studio.vectordb.collection import Collection
from embedding_studio.vectordb.collection_info_cache import CollectionInfoCache
from embedding_studio.vectordb.exceptions import CollectionNotFoundError
from embedding_studio.vectordb.pgvector.db_model import make_db_model

logger = logging.getLogger(__name__)


class PgvectorCollection(Collection):
    def __init__(
        self,
        pg_database: sqlalchemy.Engine,
        collection_id: str,
        collection_info_cache: CollectionInfoCache,
    ):
        collection_info = collection_info_cache.get_collection(collection_id)
        if not collection_info:
            raise CollectionNotFoundError(collection_id)
        self._collection_id = collection_id
        self._collection_info_cache = collection_info_cache
        self.DbModel = make_db_model(collection_info)
        self._pg_database = pg_database
        self.Session = sqlalchemy.orm.sessionmaker(pg_database)

    def get_state_info(self) -> CollectionStateInfo:
        return self._collection_info_cache.get_collection(self._collection_id)

    def insert(self, objects: List[Object]) -> None:
        with self.Session() as session:
            db_object_parts = self.DbModel.objects_to_db(objects)
            logger.debug(f"db_object_parts to insert: {db_object_parts}")
            insert_st = self.DbModel.insert_statement(db_object_parts)
            logger.debug(f"insert statement: {insert_st}")
            session.execute(insert_st)
            session.commit()

    def create_index(self) -> None:
        index = self.DbModel.hnsw_index()
        index.create(self._pg_database, checkfirst=True)
        self._collection_info_cache.set_index_state(
            self._collection_id, created=True
        )

    def upsert(self, objects: List[Object], shrink_parts: bool = True) -> None:
        with self.Session() as session:
            db_object_parts = self.DbModel.objects_to_db(objects)
            logger.debug(f"db_object_parts to upsert: {db_object_parts}")
            if shrink_parts:
                object_ids = [obj.object_id for obj in objects]
                session.execute(self.DbModel.delete_statement(object_ids))
                session.execute(self.DbModel.insert_statement(db_object_parts))
            else:
                session.execute(self.DbModel.upsert_statement(db_object_parts))
            session.commit()

    def delete(self, object_ids: List[str]) -> None:
        with self.Session() as session:
            session.execute(self.DbModel.delete_statement(object_ids))
            session.commit()

    def find_by_ids(self, object_ids: List[str]) -> List[Object]:
        with self.Session() as session:
            rows = session.execute(
                self.DbModel.find_by_id_statement(object_ids)
            )
            return self.DbModel.objects_from_db(rows)

    def find_similarities(
        self,
        query_vector: List[float],
        limit: int,
        offset: Optional[int] = None,
        max_distance: Optional[float] = None,
        payload_filter: Optional[PayloadFilter] = None,
    ) -> SearchResults:
        with self.Session() as session:
            search_st = self.DbModel.similarity_search_statement(
                query_vector=query_vector,
                limit=limit,
                offset=offset,
                max_distance=max_distance,
                payload_filter=payload_filter,
            )
            logger.debug(f"Search statement: {search_st}")
            rows = session.execute(search_st)
            found_objects = self.DbModel.similar_objects_from_db(rows)
            logger.debug(f"found db_object_parts: {found_objects}")
            next_offset: Optional[int] = None
            if len(found_objects) == limit:
                next_offset = limit + (offset or 0)
            return SearchResults(
                found_objects=found_objects,
                next_offset=next_offset,
            )

    def find_by_payload_filter(
        self,
        payload_filter: PayloadFilter,
        limit: int,
        offset: Optional[int] = None,
    ) -> SearchResults:
        with self.Session() as session:
            search_st = self.DbModel.payload_search_statement(
                payload_filter=payload_filter,
                limit=limit,
                offset=offset,
            )
            logger.debug(f"Search statement: {search_st}")
            rows = session.execute(search_st)
            found_objects = self.DbModel.found_objects_from_db(rows)
            logger.debug(f"found db_object_parts: {found_objects}")
            next_offset: Optional[int] = None
            if len(found_objects) == limit:
                next_offset = limit + (offset or 0)
            return SearchResults(
                found_objects=found_objects,
                next_offset=next_offset,
            )
