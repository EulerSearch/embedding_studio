import logging
import time
from contextlib import contextmanager
from typing import Any, List, Optional, Tuple

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
from embedding_studio.models.sort_by.models import SortByOptions
from embedding_studio.utils.dot_object import DotDict
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


import contextlib
import cProfile
import io
import pstats


@contextlib.contextmanager
def profiled():
    pr = cProfile.Profile()
    pr.enable()
    yield
    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats()
    # uncomment this to see who's calling what
    # ps.print_callers()
    print(s.getvalue())


class PgvectorCollection(Collection):
    """
    PostgreSQL vector database collection implementation.

    This class provides operations for managing vector embeddings in a PostgreSQL
    database with the pgvector extension. It supports vector similarity search,
    CRUD operations, and payload filtering.
    """

    def __init__(
        self,
        pg_database: sqlalchemy.Engine,
        collection_id: str,
        collection_info_cache: CollectionInfoCache,
    ):
        """
        Initialize the pgvector collection.

        :param pg_database: SQLAlchemy engine for PostgreSQL database connection
        :param collection_id: Unique identifier for the collection
        :param collection_info_cache: Cache for collection metadata
        :raises CollectionNotFoundError: If the collection does not exist in the cache
        """
        collection_info = collection_info_cache.get_collection(collection_id)
        if not collection_info:
            raise CollectionNotFoundError(collection_id)
        self._collection_id = collection_id
        self._collection_info_cache = collection_info_cache
        self.DbObject, self.DbObjectPart = make_db_model(collection_info)

        self._pg_database = pg_database
        self.Session = sqlalchemy.orm.sessionmaker(pg_database)

        # Create a long-lived connection for read-only operations
        self._read_connection = pg_database.connect()
        # Create a Session tied to this connection
        self._read_session = self.Session(bind=self._read_connection)

    def __del__(self):
        """Clean up resources when the object is garbage collected"""
        try:
            if hasattr(self, "_read_session"):
                self._read_session.close()
            if hasattr(self, "_read_connection"):
                self._read_connection.close()
        except Exception as e:
            logger.error(f"Error closing database connections: {e}")

    def get_info(self) -> CollectionInfo:
        """
        Get the collection metadata.

        :return: CollectionInfo object containing metadata about the collection
        """
        return self._collection_info_cache.get_collection(self._collection_id)

    def get_state_info(self) -> CollectionStateInfo:
        """
        Get the collection state information.

        :return: CollectionStateInfo object containing state information about the collection
        """
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
        """
        Insert objects with their vector parts into the collection.

        :param objects: List of Object instances to insert
        """
        db_objects = [
            self.DbObject(
                object_id=obj.object_id,
                payload=obj.payload,
                storage_meta=obj.storage_meta,
                user_id=obj.user_id,
                original_id=obj.original_id,
                session_id=obj.session_id,
            )
            for obj in objects
        ]
        db_parts = [
            self.DbObjectPart(
                object_id=obj.object_id,
                part_id=part.part_id,
                vector=part.vector,
                object=db_objects[i],
                is_average=part.is_average,
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
        """
        Create a vector index for the collection.

        Creates an HNSW index on the vector column of the object parts table
        and updates the collection's index state.
        """
        index = self.DbObjectPart.hnsw_index()
        index.create(self._pg_database, checkfirst=True)
        self._collection_info_cache.set_index_state(
            self._collection_id, created=True
        )

    def upsert(self, objects: List[Object], shrink_parts: bool = True) -> None:
        """
        Update or insert objects with their vector parts.

        :param objects: List of Object instances to upsert
        :param shrink_parts: If True, delete existing parts before inserting new ones;
                            if False, perform an actual upsert on parts
        """
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
                is_average=part.is_average,
            )
            for i, obj in enumerate(objects)
            for part in obj.parts
        ]

        with self.Session() as session, session.begin():
            logger.info("Session obtained")
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
        """
        Delete objects and their parts from the collection.

        :param object_ids: List of object IDs to delete
        """
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

    def _reset_read_session(self):
        """
        Reconnect and reset the persistent read session.

        :return: A new SQLAlchemy session
        """
        self._read_connection = self._pg_database.connect()
        self._read_session = self.Session(bind=self._read_connection)
        return self._read_session

    def _with_read_session(self, query_func):
        """
        Executes the provided query_func using the persistent read session.

        Falls back to a traditional session in case of failure.

        :param query_func: Function that takes a session parameter and performs queries
        :return: Result of the query_func
        """
        try:
            if self._read_connection.closed:
                self._reset_read_session()
            return query_func(self._read_session)
        except Exception as e:
            logger.error(f"Error in persistent session: {e}")
            # Fallback to traditional session
            with self.Session() as session:
                return query_func(session)

    def find_by_ids(self, object_ids: List[str]) -> List[Object]:
        """
        Find objects by their IDs.

        :param object_ids: List of object IDs to find
        :return: List of Object instances
        """

        def query(session):
            rows = session.execute(
                self.DbObjectPart.find_by_id_statement(object_ids)
            )
            return self.DbObjectPart.objects_from_db(rows)

        return self._with_read_session(query)

    def find_by_original_ids(self, object_ids: List[str]) -> List[Object]:
        """
        Find objects by their original IDs.

        :param object_ids: List of original object IDs to find
        :return: List of Object instances
        """

        def query(session):
            rows = session.execute(
                self.DbObjectPart.find_by_original_id_statement(object_ids)
            )
            return self.DbObjectPart.objects_from_db(rows)

        return self._with_read_session(query)

    def get_total(self, originals_only: bool = True) -> int:
        """
        Get the total number of objects in the collection.

        :param originals_only: If True, count only original objects (not derivatives)
        :return: Total number of objects
        """

        def query(session):
            total = session.execute(
                self.DbObject.get_total_statement(originals_only)
            ).scalar_one()
            return int(total)

        return self._with_read_session(query)

    def get_objects_common_data_batch(
        self,
        limit: int,
        offset: Optional[int] = None,
        originals_only: bool = True,
    ) -> ObjectsCommonDataBatch:
        """
        Retrieve common data for a batch of objects.

        :param limit: Maximum number of objects to retrieve
        :param offset: Number of objects to skip
        :param originals_only: If True, retrieve only original objects (not derivatives)
        :return: ObjectsCommonDataBatch containing object data and pagination info
        """

        def query(session):
            total = int(
                session.execute(
                    self.DbObject.get_total_statement(originals_only)
                ).scalar_one()
            )
            next_offset = (
                (offset + limit)
                if offset is not None and (offset + limit < total)
                else None
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

        return self._with_read_session(query)

    def find_similarities(
        self,
        query_vector: List[float],
        limit: int,
        offset: Optional[int] = None,
        max_distance: Optional[float] = None,
        payload_filter: Optional[PayloadFilter] = None,
        sort_by: Optional[SortByOptions] = None,
        user_id: Optional[str] = None,
        similarity_first: bool = False,
        meta_info: Any = None,
    ) -> SearchResults:
        """
        Find objects similar to the query vector.

        :param query_vector: Vector to compare against
        :param limit: Maximum number of objects to retrieve
        :param offset: Number of objects to skip
        :param max_distance: Maximum distance threshold for similarity
        :param payload_filter: Filter to apply on object payloads
        :param sort_by: Sorting options
        :param user_id: Filter objects by user ID
        :param similarity_first: If True, sort by similarity first, then by sort_by field
        :param meta_info: Additional metadata for the query
        :return: SearchResults object containing similar objects and pagination info
        """

        def query(session):
            search_st = self.DbObjectPart.similarity_search_statement(
                query_vector=query_vector,
                limit=limit,
                offset=offset,
                max_distance=max_distance,
                payload_filter=payload_filter,
                sort_by=sort_by,
                user_id=user_id,
                similarity_first=similarity_first,
                meta_info=meta_info,
            )
            result = session.execute(search_st)
            rows = [DotDict(dict(row._mapping)) for row in result]
            subset_count = 0
            if rows and len(rows) > 0:
                subset_count = rows[0].subset_count

            found_objects = self.DbObjectPart.similar_objects_from_db(rows)
            next_offset: Optional[int] = None
            if len(found_objects) == limit:
                next_offset = limit + (offset or 0)
            return SearchResults(
                found_objects=found_objects,
                next_offset=next_offset,
                meta_info={"subset_count": subset_count},
            )

        return self._with_read_session(query)

    def find_similar_objects(
        self,
        query_vector: List[float],
        limit: int,
        offset: Optional[int] = None,
        max_distance: Optional[float] = None,
        payload_filter: Optional[PayloadFilter] = None,
        sort_by: Optional[SortByOptions] = None,
        user_id: Optional[str] = None,
        with_vectors: bool = True,
        similarity_first: bool = False,
        meta_info: Any = None,
    ) -> Tuple[List[ObjectWithDistance], Any]:
        """
        Find objects similar to the query vector with distance information.

        :param query_vector: Vector to compare against
        :param limit: Maximum number of objects to retrieve
        :param offset: Number of objects to skip
        :param max_distance: Maximum distance threshold for similarity
        :param payload_filter: Filter to apply on object payloads
        :param sort_by: Sorting options
        :param user_id: Filter objects by user ID
        :param with_vectors: If True, include vectors in the results
        :param similarity_first: If True, sort by similarity first, then by sort_by field
        :param meta_info: Additional metadata for the query
        :return: Tuple of (List of ObjectWithDistance instances, meta_info dictionary)
        """

        def query(session):
            search_st = self.DbObjectPart.similarity_search_statement(
                query_vector=query_vector,
                limit=limit,
                offset=offset,
                max_distance=max_distance,
                payload_filter=payload_filter,
                sort_by=sort_by,
                user_id=user_id,
                with_vectors=with_vectors,
                similarity_first=similarity_first,
                meta_info=meta_info,
            )
            result = session.execute(search_st)
            rows = [DotDict(dict(row._mapping)) for row in result]
            subset_count = 0
            if rows and len(rows) > 0:
                subset_count = rows[0].subset_count

            found_objects = self.DbObjectPart.objects_with_distance_from_db(
                rows
            )
            return found_objects, {"subset_count": subset_count}

        return self._with_read_session(query)

    def find_by_payload_filter(
        self,
        payload_filter: PayloadFilter,
        limit: int,
        offset: Optional[int] = None,
        sort_by: Optional[SortByOptions] = None,
    ) -> SearchResults:
        """
        Find objects matching a payload filter.

        :param payload_filter: Filter to apply on object payloads
        :param limit: Maximum number of objects to retrieve
        :param offset: Number of objects to skip
        :param sort_by: Sorting options
        :return: SearchResults object containing found objects and pagination info
        """

        def query(session):
            search_st = self.DbObjectPart.payload_search_statement(
                payload_filter=payload_filter,
                sort_by=sort_by,
                limit=limit,
                offset=offset,
            )
            rows = session.execute(search_st).all()
            found_objects = self.DbObjectPart.found_objects_from_db(rows)
            next_offset: Optional[int] = None
            if len(found_objects) == limit:
                next_offset = limit + (offset or 0)

            return SearchResults(
                found_objects=found_objects,
                next_offset=next_offset,
                total_count=None,
            )

        return self._with_read_session(query)

    def count_by_payload_filter(self, payload_filter: PayloadFilter) -> int:
        """
        Count objects matching a payload filter.

        :param payload_filter: Filter to apply on object payloads
        :return: Count of matching objects
        """

        def query(session):
            count_st = self.DbObjectPart.payload_count_statement(
                payload_filter=payload_filter,
            )
            total_count = session.execute(count_st).scalar()
            return total_count

        return self._with_read_session(query)


class PgvectorQueryCollection(PgvectorCollection, QueryCollection):
    """
    PostgreSQL vector database query collection implementation.

    This class extends PgvectorCollection with additional query-specific functionality.
    It's designed for handling user queries and related operations.
    """

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
