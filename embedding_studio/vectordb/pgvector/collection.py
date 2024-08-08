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
        self.DbObject, self.DbObjectPart = make_db_model(collection_info)
        self._pg_database = pg_database
        self.Session = sqlalchemy.orm.sessionmaker(pg_database)

    def get_state_info(self) -> CollectionStateInfo:
        return self._collection_info_cache.get_collection(self._collection_id)

    def insert(self, objects: List[Object]) -> None:
        db_objects = [
            self.DbObject(
                object_id=obj.object_id,
                payload=obj.payload,
                storage_meta=obj.storage_meta,
            )
            for obj in objects
        ]
        db_parts = [
            self.DbObjectPart(
                object_id=obj.object_id,
                part_id=part.part_id,
                vector=part.vector,
                object=db_objects[i],
            )
            for i, obj in enumerate(objects)
            for part in obj.parts
        ]

        with self.Session() as session, session.begin():
            try:
                # Insert objects
                insert_st = self.DbObject.insert_objects_statement(db_objects)
                session.execute(insert_st)

                # Insert parts
                insert_st = self.DbObjectPart.insert_parts_statement(db_parts)
                session.execute(insert_st)

            except Exception as e:
                logger.error(f"Failed to insert objects with parts: {e}")
                raise

    def create_index(self) -> None:
        index = self.DbObjectPart.hnsw_index()
        index.create(self._pg_database, checkfirst=True)
        self._collection_info_cache.set_index_state(
            self._collection_id, created=True
        )

    def upsert(self, objects: List[Object], shrink_parts: bool = True) -> None:
        db_objects = [
            self.DbObject(
                object_id=obj.object_id,
                payload=obj.payload,
                storage_meta=obj.storage_meta,
            )
            for obj in objects
        ]
        db_parts = [
            self.DbObjectPart(
                object_id=obj.object_id,
                part_id=part.part_id,
                vector=part.vector,
                object=db_objects[i],
            )
            for i, obj in enumerate(objects)
            for part in obj.parts
        ]

        with self.Session() as session, session.begin():
            try:
                # Upsert objects
                upsert_st = self.DbObject.upsert_objects_statement(db_objects)
                session.execute(upsert_st)

                if shrink_parts:
                    # Get object IDs
                    object_ids = [obj.object_id for obj in objects]

                    # Delete old parts
                    delete_parts_st = self.DbObjectPart.delete_statement(
                        object_ids
                    )
                    session.execute(delete_parts_st)

                    # Insert new parts
                    insert_parts_st = self.DbObjectPart.insert_parts_statement(
                        db_parts
                    )
                    session.execute(insert_parts_st)
                else:
                    # Upsert parts without deletion
                    upsert_parts_st = self.DbObjectPart.upsert_parts_statement(
                        db_parts
                    )
                    session.execute(upsert_parts_st)

            except Exception as e:
                logger.exception(f"Failed to upsert objects with parts: {e}")
                raise

    def delete(self, object_ids: List[str]) -> None:
        with self.Session() as session, session.begin():
            session.execute(self.DbObjectPart.delete_statement(object_ids))
            session.execute(self.DbObject.delete_statement(object_ids))

    def find_by_ids(self, object_ids: List[str]) -> List[Object]:
        with self.Session() as session, session.begin():
            rows = session.execute(
                self.DbObject.find_by_id_statement(object_ids)
            )
            return self.DbObject.objects_from_db(rows)

    def find_similarities(
        self,
        query_vector: List[float],
        limit: int,
        offset: Optional[int] = None,
        max_distance: Optional[float] = None,
        payload_filter: Optional[PayloadFilter] = None,
    ) -> SearchResults:
        with self.Session() as session, session.begin():
            search_st = self.DbObjectPart.similarity_search_statement(
                query_vector=query_vector,
                limit=limit,
                offset=offset,
                max_distance=max_distance,
                payload_filter=payload_filter,
            )
            logger.debug(f"Search statement: {search_st}")
            rows = session.execute(search_st)
            found_objects = self.DbObjectPart.similar_objects_from_db(rows)
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
        with self.Session() as session, session.begin():
            search_st = self.DbObjectPart.payload_search_statement(
                payload_filter=payload_filter,
                limit=limit,
                offset=offset,
            )
            logger.debug(f"Search statement: {search_st}")
            rows = session.execute(search_st)
            found_objects = self.DbObjectPart.found_objects_from_db(rows)
            logger.debug(f"found db_object_parts: {found_objects}")
            next_offset: Optional[int] = None
            if len(found_objects) == limit:
                next_offset = limit + (offset or 0)
            return SearchResults(
                found_objects=found_objects,
                next_offset=next_offset,
            )
