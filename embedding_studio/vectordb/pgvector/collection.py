import logging
import time
from contextlib import contextmanager
from typing import List, Optional

import sqlalchemy
from sqlalchemy.exc import OperationalError, SQLAlchemyError

from embedding_studio.models.embeddings.collections import CollectionStateInfo
from embedding_studio.models.embeddings.objects import (
    Object,
    ObjectsCommonDataBatch,
    ObjectWithDistance,
    SearchResults,
)
from embedding_studio.models.payload.models import PayloadFilter
from embedding_studio.vectordb.collection import Collection, QueryCollection
from embedding_studio.vectordb.collection_info_cache import (
    CollectionInfo,
    CollectionInfoCache,
)
from embedding_studio.vectordb.exceptions import (
    CollectionNotFoundError,
    LockAcquisitionError,
)
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

    def get_info(self) -> CollectionInfo:
        return self._collection_info_cache.get_collection(self._collection_id)

    def get_state_info(self) -> CollectionStateInfo:
        return self._collection_info_cache.get_collection(self._collection_id)

    @contextmanager
    def lock_objects(
        self,
        object_ids: List[str],
        max_attempts: int = 5,
        wait_time: float = 1.0,
    ):
        """
        Context manager to lock the specified objects within a transaction.

        :param object_ids: List of object IDs to lock.
        :param max_attempts: Maximum number of attempts to acquire the lock.
        :param wait_time: Time to wait between attempts (in seconds).
        """
        attempt = 0

        with self.Session() as session:
            session.begin()  # Begin transaction

            while attempt < max_attempts:
                try:
                    # Try to lock the objects
                    lock_statement = sqlalchemy.text(
                        f'SELECT 1 FROM "{self.DbObject.__tablename__}" '
                        f"WHERE object_id = ANY(:object_ids) FOR UPDATE NOWAIT"
                    ).bindparams(object_ids=object_ids)
                    session.execute(lock_statement)
                    break  # Lock acquired successfully

                except OperationalError as e:
                    if "55P03" in str(
                        e.orig.pgcode
                    ):  # "55P03" is the lock not available error code
                        attempt += 1
                        if attempt >= max_attempts:
                            logger.error(
                                f"Failed to obtain lock for objects after {max_attempts} attempts: {object_ids}"
                            )
                            raise LockAcquisitionError(
                                f"Could not obtain lock for objects: {object_ids}"
                            )
                        logger.warning(
                            f"Could not obtain lock, attempt {attempt}/{max_attempts}. Waiting {wait_time} seconds before retry."
                        )
                        time.sleep(wait_time)
                    else:
                        raise
                except SQLAlchemyError as e:
                    logger.exception(f"Unexpected SQLAlchemy error: {e}")
                    session.rollback()
                    raise

            # Yield control back to the caller while holding the lock
            try:
                yield session

                # Commit the transaction after the operations
                session.commit()

            except Exception as e:
                logger.exception(f"Error during operation: {e}")
                session.rollback()
                raise

    def insert(self, objects: List[Object]) -> None:
        db_objects = [
            self.DbObject(
                object_id=obj.object_id,
                payload=obj.payload,
                storage_meta=obj.storage_meta,
                user_id=obj.user_id,
                original_id=obj.original_id,
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
                user_id=obj.user_id,
                original_id=obj.original_id,
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
            try:
                # Delete from DbObjectPart first to avoid deadlocks
                session.execute(self.DbObjectPart.delete_statement(object_ids))

                # Then delete from DbObject
                session.execute(self.DbObject.delete_statement(object_ids))

            except Exception as e:
                logger.error(f"Failed to delete objects: {e}")
                session.rollback()
                raise
            else:
                session.commit()

    def find_by_ids(self, object_ids: List[str]) -> List[Object]:
        with self.Session() as session, session.begin():
            rows = session.execute(
                self.DbObjectPart.find_by_id_statement(object_ids)
            )
            return self.DbObjectPart.objects_from_db(rows)

    def find_by_original_ids(self, object_ids: List[str]) -> List[Object]:
        with self.Session() as session, session.begin():
            rows = session.execute(
                self.DbObjectPart.find_by_original_id_statement(object_ids)
            )
            return self.DbObjectPart.objects_from_db(rows)

    def get_total(self, originals_only: bool = True) -> int:
        with self.Session() as session, session.begin():
            total = session.execute(
                self.DbObject.get_total_statement(originals_only)
            ).scalar_one()

        return int(total)

    def get_objects_common_data_batch(
        self,
        limit: int,
        offset: Optional[int] = None,
        originals_only: bool = True,
    ) -> ObjectsCommonDataBatch:
        with self.Session() as session, session.begin():
            total = int(
                session.execute(
                    self.DbObject.get_total_statement(originals_only)
                ).scalar_one()
            )

            next_offset = None
            if offset is not None:
                next_offset = (
                    (offset + limit) if (offset + limit < total) else None
                )

            data = session.execute(
                self.DbObject.get_objects_common_data_batch_statement(
                    limit, offset, originals_only
                )
            ).all()
            objects_info = self.DbObject.objects_common_data_from_db(data)

        return ObjectsCommonDataBatch(
            objects_info=objects_info, total=total, next_offset=next_offset
        )

    def find_similarities(
        self,
        query_vector: List[float],
        limit: int,
        offset: Optional[int] = None,
        max_distance: Optional[float] = None,
        payload_filter: Optional[PayloadFilter] = None,
        user_id: Optional[str] = None,
    ) -> SearchResults:
        with self.Session() as session, session.begin():
            search_st = self.DbObjectPart.similarity_search_statement(
                query_vector=query_vector,
                limit=limit,
                offset=offset,
                max_distance=max_distance,
                payload_filter=payload_filter,
                user_id=user_id,
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

    def find_similar_objects(
        self,
        query_vector: List[float],
        limit: int,
        offset: Optional[int] = None,
        max_distance: Optional[float] = None,
        payload_filter: Optional[PayloadFilter] = None,
        user_id: Optional[str] = None,
    ) -> List[ObjectWithDistance]:
        with self.Session() as session, session.begin():
            search_st = self.DbObjectPart.similarity_search_statement(
                query_vector=query_vector,
                limit=limit,
                offset=offset,
                max_distance=max_distance,
                payload_filter=payload_filter,
                user_id=user_id,
                with_vectors=True,
            )
            logger.debug(f"Search statement: {search_st}")
            rows = session.execute(search_st)
            found_objects = self.DbObjectPart.objects_with_distance_from_db(rows)

            return found_objects

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


class PgvectorQueryCollection(PgvectorCollection, QueryCollection):
    def get_objects_by_session_id(self, session_id: str) -> List[Object]:
        """
        Retrieve objects and their parts by session ID with vector validation.

        :param session_id: The session ID to query.
        :return: List of Object instances with their parts.
        """
        with self.Session() as session, session.begin():
            try:
                query = self.DbObjectPart.find_by_session_id_statement(
                    session_id
                )
                rows = session.execute(query).all()

                if not rows:
                    logger.warning(
                        f"No objects found for session ID: {session_id}"
                    )
                    return []

                objects = self.DbObjectPart.objects_from_db(rows)
                return objects

            except Exception as e:
                logger.exception(
                    f"Failed to fetch objects by session ID {session_id}: {e}"
                )
                raise
