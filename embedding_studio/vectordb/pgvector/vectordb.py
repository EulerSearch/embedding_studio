from typing import List, Optional

import pymongo
import sqlalchemy
from sqlalchemy import text

from embedding_studio.models.embeddings.collections import (
    CollectionInfo,
    CollectionStateInfo,
    CollectionWorkState,
)
from embedding_studio.models.embeddings.models import EmbeddingModelInfo
from embedding_studio.vectordb.collection import Collection, QueryCollection
from embedding_studio.vectordb.collection_info_cache import CollectionInfoCache
from embedding_studio.vectordb.exceptions import (
    CollectionNotFoundError,
    CreateCollectionConflictError,
    DeleteBlueCollectionError,
)
from embedding_studio.vectordb.optimization import Optimization
from embedding_studio.vectordb.pgvector.collection import (
    PgvectorCollection,
    PgvectorQueryCollection,
)
from embedding_studio.vectordb.pgvector.db_model import make_db_model
from embedding_studio.vectordb.pgvector.functions.advanced_similarity import (
    generate_advanced_vector_search_function,
    generate_advanced_vector_search_no_vectors_function,
    generate_advanced_vector_search_similarity_ordered_function,
    generate_advanced_vector_search_similarity_ordered_no_vectors_function,
)
from embedding_studio.vectordb.pgvector.functions.simple_similarity import (
    generate_simple_vector_search_function,
    generate_simple_vector_search_no_vectors_function,
    generate_simple_vector_search_similarity_ordered_function,
    generate_simple_vector_search_similarity_ordered_no_vectors_function,
)
from embedding_studio.vectordb.vectordb import VectorDb


class PgvectorDb(VectorDb):
    """
    PostgreSQL vector database implementation using pgvector extension.

    This class provides vector database operations using PostgreSQL with the pgvector
    extension for efficient vector similarity search and storage.
    """

    def __init__(
        self,
        pg_database: sqlalchemy.Engine,
        embeddings_mongo_database: pymongo.database.Database,
        prefix: str = "basic",
        optimizations: Optional[List[Optimization]] = None,
        query_optimizations: Optional[List[Optimization]] = None,
    ):
        """
        Initialize the PostgreSQL vector database.

        :param pg_database: SQLAlchemy engine for PostgreSQL database connection
        :param embeddings_mongo_database: MongoDB database for storing collection metadata
        :param prefix: Prefix for database identifier
        :param optimizations: List of optimization strategies to apply to collections
        :param query_optimizations: List of optimization strategies to apply to query collections
        """
        super(PgvectorDb, self).__init__(optimizations, query_optimizations)
        db_id: str = f"{prefix}_pgvector_single_db"
        self._pg_database = pg_database
        self._collection_info_cache = CollectionInfoCache(
            mongo_database=embeddings_mongo_database,
            db_id=db_id,
        )
        self._init_pgvector()

    def _init_pgvector(self):
        """
        Initialize the pgvector extension in PostgreSQL.

        Ensures the vector extension is created in the PostgreSQL database.
        """
        with self._pg_database.begin() as connection:
            connection.execute(
                sqlalchemy.text("CREATE EXTENSION IF NOT EXISTS vector")
            )

    def update_info(self):
        """
        Update internal information about collections by invalidating the cache.

        Forces a refresh of collection metadata from the database.
        """
        self._collection_info_cache.invalidate_cache()

    def list_collections(self) -> List[CollectionStateInfo]:
        """
        List all available collections in the database.

        :return: List of CollectionStateInfo objects representing available collections
        """
        return self._collection_info_cache.list_collections()

    def list_query_collections(self) -> List[CollectionStateInfo]:
        """
        List all available query collections in the database.

        :return: List of CollectionStateInfo objects representing available query collections
        """
        return self._collection_info_cache.list_query_collections()

    def get_collection(
        self,
        embedding_model_id: str,
    ) -> Collection:
        """
        Retrieve a pgvector collection by its embedding model ID.

        :param embedding_model_id: The ID of the embedding model associated with the collection
        :return: A PgvectorCollection object for the specified embedding model
        """
        return PgvectorCollection(
            pg_database=self._pg_database,
            collection_id=embedding_model_id,
            collection_info_cache=self._collection_info_cache,
        )

    def get_query_collection(
        self,
        embedding_model_id,
    ) -> QueryCollection:
        """
        Retrieve a pgvector query collection by its embedding model ID.

        :param embedding_model_id: The ID of the embedding model associated with the query collection
        :return: A PgvectorQueryCollection object for the specified embedding model
        """
        return PgvectorQueryCollection(
            pg_database=self._pg_database,
            collection_id=self.get_query_collection_id(embedding_model_id),
            collection_info_cache=self._collection_info_cache,
        )

    def get_blue_collection(self) -> Optional[Collection]:
        """
        Get the current "blue" (active/primary) collection.

        :return: The active PgvectorCollection object or None if no blue collection exists
        """
        info = self._collection_info_cache.get_blue_collection()
        if info:
            return self.get_collection(info.embedding_model.id)
        return None

    def get_blue_query_collection(self) -> Optional[QueryCollection]:
        """
        Get the current "blue" (active/primary) query collection.

        :return: The active PgvectorQueryCollection object or None if no blue query collection exists
        """
        info = self._collection_info_cache.get_blue_query_collection()
        if info:
            return self.get_query_collection(info.embedding_model.id)
        return None

    def set_blue_collection(
        self,
        embedding_model_id: str,
    ) -> None:
        """
        Set a collection as the "blue" (active/primary) collection.

        :param embedding_model_id: The ID of the embedding model associated with the collection
        """
        collection_id = embedding_model_id
        query_collection_id = self.get_query_collection_id(embedding_model_id)
        self._collection_info_cache.set_blue_collection(
            collection_id,
            query_collection_id
            if self.query_collection_exists(collection_id)
            else None,
        )

    def save_collection_info(self, collection_info: CollectionInfo):
        """
        Save or update collection information in the metadata store.

        :param collection_info: The CollectionInfo object to save
        """
        self._collection_info_cache.update_collection(collection_info)

    def save_query_collection_info(self, collection_info: CollectionInfo):
        """
        Save or update query collection information in the metadata store.

        :param collection_info: The CollectionInfo object to save
        """
        self._collection_info_cache.update_query_collection(collection_info)

    # TODO: decided to think about potential functions merging later
    def _create_collection(
        self,
        embedding_model: EmbeddingModelInfo,
    ) -> Collection:
        """
        Internal method to create a new pgvector collection.

        Creates the necessary tables and indexes in PostgreSQL and registers
        the collection in the metadata store. Also creates SQL functions for
        vector search operations.

        :param embedding_model: The EmbeddingModelInfo object representing the model for this collection
        :return: A newly created PgvectorCollection object
        """
        collection_info = CollectionInfo(
            collection_id=embedding_model.id,
            embedding_model=embedding_model,
        )
        db_object_model, db_object_part_model = make_db_model(collection_info)
        db_object_model.create_table(self._pg_database)
        db_object_part_model.create_table(self._pg_database)

        db_object_part_model.hnsw_index().create(
            self._pg_database, checkfirst=True
        )

        # TODO: protect from race condition
        # TODO: protect from inconsistent state (after crash at this point)
        created_collection_info = self._collection_info_cache.add_collection(
            collection_info
        )

        simple_vector_search_similarity_ordered_function = text(
            generate_simple_vector_search_similarity_ordered_function(
                embedding_model.id, metric_type=embedding_model.metric_type
            )
        )
        simple_vector_search_similarity_ordered_no_vectors_function = text(
            generate_simple_vector_search_similarity_ordered_no_vectors_function(
                embedding_model.id, metric_type=embedding_model.metric_type
            )
        )

        simple_vector_search_function = text(
            generate_simple_vector_search_function(
                embedding_model.id, metric_type=embedding_model.metric_type
            )
        )
        simple_vector_search_no_vectors_function = text(
            generate_simple_vector_search_no_vectors_function(
                embedding_model.id, metric_type=embedding_model.metric_type
            )
        )

        advanced_vector_search_similarity_ordered_function = text(
            generate_advanced_vector_search_similarity_ordered_function(
                embedding_model.id, metric_type=embedding_model.metric_type
            )
        )
        advanced_vector_search_similarity_ordered_no_vectors_function = text(
            generate_advanced_vector_search_similarity_ordered_no_vectors_function(
                embedding_model.id, metric_type=embedding_model.metric_type
            )
        )

        advanced_vector_search_function = text(
            generate_advanced_vector_search_function(
                embedding_model.id, metric_type=embedding_model.metric_type
            )
        )
        advanced_vector_search_no_vectors_function = text(
            generate_advanced_vector_search_no_vectors_function(
                embedding_model.id, metric_type=embedding_model.metric_type
            )
        )

        with self._pg_database.begin() as connection:
            connection.execute(
                simple_vector_search_similarity_ordered_function
            )
            connection.execute(
                simple_vector_search_similarity_ordered_no_vectors_function
            )

            connection.execute(simple_vector_search_function)
            connection.execute(simple_vector_search_no_vectors_function)

            connection.execute(
                advanced_vector_search_similarity_ordered_function
            )
            connection.execute(
                advanced_vector_search_similarity_ordered_no_vectors_function
            )

            connection.execute(advanced_vector_search_function)
            connection.execute(advanced_vector_search_no_vectors_function)

        if created_collection_info.embedding_model != embedding_model:
            raise CreateCollectionConflictError(
                model_passed=embedding_model,
                model_used=collection_info.embedding_model,
            )
        return self.get_collection(embedding_model.id)

    def _create_query_collection(
        self,
        embedding_model: EmbeddingModelInfo,
    ) -> QueryCollection:
        """
        Internal method to create a new pgvector query collection.

        Creates the necessary tables and indexes in PostgreSQL and registers
        the query collection in the metadata store.

        :param embedding_model: The EmbeddingModelInfo object representing the model for this query collection
        :return: A newly created PgvectorQueryCollection object
        """
        collection_info = CollectionInfo(
            collection_id=self.get_query_collection_id(embedding_model.id),
            embedding_model=embedding_model,
        )
        db_object_model, db_object_part_model = make_db_model(collection_info)
        db_object_model.create_table(self._pg_database)
        db_object_part_model.create_table(self._pg_database)

        # TODO: protect from race condition
        # TODO: protect from inconsistent state (after crash at this point)
        created_collection_info = (
            self._collection_info_cache.add_query_collection(collection_info)
        )
        if created_collection_info.embedding_model != embedding_model:
            raise CreateCollectionConflictError(
                model_passed=embedding_model,
                model_used=collection_info.embedding_model,
            )
        return self.get_query_collection(embedding_model.id)

    def collection_exists(self, embedding_model_id: str) -> bool:
        """
        Check if a collection exists for the given embedding model ID.

        :param embedding_model_id: The ID of the embedding model to check
        :return: True if the collection exists, False otherwise
        """
        collection_info = self._collection_info_cache.get_collection(
            collection_id=embedding_model_id
        )
        return collection_info is not None

    def query_collection_exists(self, embedding_model_id: str) -> bool:
        """
        Check if a query collection exists for the given embedding model ID.

        :param embedding_model_id: The ID of the embedding model to check
        :return: True if the query collection exists, False otherwise
        """
        collection_info = self._collection_info_cache.get_collection(
            self.get_query_collection_id(embedding_model_id)
        )
        return collection_info is not None

    def delete_collection(self, embedding_model_id: str) -> None:
        """
        Delete a collection and its associated database objects.

        :param embedding_model_id: The ID of the embedding model associated with the collection to delete
        :raises CollectionNotFoundError: If the collection does not exist
        :raises DeleteBlueCollectionError: If attempting to delete the active "blue" collection
        """
        col_info = self._collection_info_cache.get_collection(
            embedding_model_id
        )
        if not col_info:
            raise CollectionNotFoundError(embedding_model_id)
        if col_info.work_state == CollectionWorkState.BLUE:
            raise DeleteBlueCollectionError()

        db_object_model, db_object_part_model = make_db_model(col_info)
        db_object_part_model.__table__.drop(self._pg_database, checkfirst=True)
        db_object_model.__table__.drop(self._pg_database, checkfirst=True)

        # TODO: protect from inconsistent state (after crash at this point)
        self._collection_info_cache.delete_collection(embedding_model_id)

    def delete_query_collection(self, embedding_model_id: str) -> None:
        """
        Delete a query collection and its associated database objects.

        :param embedding_model_id: The ID of the embedding model associated with the query collection to delete
        :raises CollectionNotFoundError: If the query collection does not exist
        :raises DeleteBlueCollectionError: If attempting to delete the active "blue" query collection
        """
        col_info = self._collection_info_cache.get_collection(
            self.get_query_collection_id(embedding_model_id)
        )
        if not col_info:
            raise CollectionNotFoundError(
                self.get_query_collection_id(embedding_model_id)
            )
        if col_info.work_state == CollectionWorkState.BLUE:
            raise DeleteBlueCollectionError()

        db_object_model, db_object_part_model = make_db_model(col_info)
        db_object_part_model.__table__.drop(self._pg_database, checkfirst=True)
        db_object_model.__table__.drop(self._pg_database, checkfirst=True)

        # TODO: protect from inconsistent state (after crash at this point)
        self._collection_info_cache.delete_collection(
            self.get_query_collection_id(embedding_model_id)
        )
