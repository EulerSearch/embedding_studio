from typing import List, Optional

import pymongo
import sqlalchemy

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
from embedding_studio.vectordb.pgvector.collection import (
    PgvectorCollection,
    PgvectorQueryCollection,
)
from embedding_studio.vectordb.pgvector.db_model import make_db_model
from embedding_studio.vectordb.vectordb import VectorDb


class PgvectorDb(VectorDb):
    def __init__(
        self,
        pg_database: sqlalchemy.Engine,
        embeddings_mongo_database: pymongo.database.Database,
        prefix: str = "basic",
    ):
        db_id: str = f"{prefix}_pgvector_single_db"
        self._pg_database = pg_database
        self._collection_info_cache = CollectionInfoCache(
            mongo_database=embeddings_mongo_database,
            db_id=db_id,
        )
        self._init_pgvector()

    def _init_pgvector(self):
        with self._pg_database.begin() as connection:
            connection.execute(
                sqlalchemy.text("CREATE EXTENSION IF NOT EXISTS vector")
            )

    def update_info(self):
        self._collection_info_cache.invalidate_cache()

    def list_collections(self) -> List[CollectionStateInfo]:
        return self._collection_info_cache.list_collections()

    def list_query_collections(self) -> List[CollectionStateInfo]:
        return self._collection_info_cache.list_query_collections()

    def get_collection(
        self,
        embedding_model_id: str,
    ) -> Collection:
        return PgvectorCollection(
            pg_database=self._pg_database,
            collection_id=embedding_model_id,
            collection_info_cache=self._collection_info_cache,
        )

    def get_query_collection(
        self,
        embedding_model_id,
    ) -> QueryCollection:
        return PgvectorQueryCollection(
            pg_database=self._pg_database,
            collection_id=self.get_query_collection_id(embedding_model_id),
            collection_info_cache=self._collection_info_cache,
        )

    def get_blue_collection(self) -> Optional[Collection]:
        info = self._collection_info_cache.get_blue_collection()
        if info:
            return self.get_collection(info.embedding_model.id)
        return None

    def get_blue_query_collection(self) -> Optional[QueryCollection]:
        info = self._collection_info_cache.get_blue_query_collection()
        if info:
            return self.get_query_collection(info.embedding_model.id)
        return None

    def set_blue_collection(
        self,
        embedding_model_id: str,
    ) -> None:
        collection_id = embedding_model_id
        query_collection_id = self.get_query_collection_id(embedding_model_id)
        self._collection_info_cache.set_blue_collection(
            collection_id,
            query_collection_id
            if self.query_collection_exists(collection_id)
            else None,
        )

    # TODO: decided to think about potential functions merging later
    def create_collection(
        self,
        embedding_model: EmbeddingModelInfo,
    ) -> Collection:
        collection_info = CollectionInfo(
            collection_id=embedding_model.id,
            embedding_model=embedding_model,
        )
        db_object_model, db_object_part_model = make_db_model(collection_info)
        db_object_model.create_table(self._pg_database)
        db_object_part_model.create_table(self._pg_database)

        # TODO: protect from race condition
        # TODO: protect from inconsistent state (after crash at this point)
        created_collection_info = self._collection_info_cache.add_collection(
            collection_info
        )
        if created_collection_info.embedding_model != embedding_model:
            raise CreateCollectionConflictError(
                model_passed=embedding_model,
                model_used=collection_info.embedding_model,
            )
        return self.get_collection(embedding_model.id)

    def create_query_collection(
        self,
        embedding_model: EmbeddingModelInfo,
    ) -> QueryCollection:
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
        collection_info = self._collection_info_cache.get_collection(
            collection_id=embedding_model_id
        )
        return collection_info is not None

    def query_collection_exists(self, embedding_model_id: str) -> bool:
        collection_info = self._collection_info_cache.get_collection(
            self.get_query_collection_id(embedding_model_id)
        )
        return collection_info is not None

    def delete_collection(self, embedding_model_id: str) -> None:
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
