from typing import List, Optional

import pymongo
import sqlalchemy

from embedding_studio.models.embeddings.collections import (
    CollectionInfo,
    CollectionStateInfo,
    CollectionWorkState,
)
from embedding_studio.models.embeddings.models import EmbeddingModel
from embedding_studio.vectordb.collection import Collection
from embedding_studio.vectordb.collection_info_cache import CollectionInfoCache
from embedding_studio.vectordb.exceptions import (
    CollectionNotFoundError,
    CreateCollectionConflictError,
    DeleteBlueCollectionError,
)
from embedding_studio.vectordb.pgvector.collection import PgvectorCollection
from embedding_studio.vectordb.pgvector.db_model import make_db_model
from embedding_studio.vectordb.vectordb import VectorDb


class PgvectorDb(VectorDb):
    def __init__(
        self,
        pg_database: sqlalchemy.Engine,
        embeddings_mongo_database: pymongo.database.Database,
    ):
        self._pg_database = pg_database
        self._collection_info_cache = CollectionInfoCache(
            mongo_database=embeddings_mongo_database,
            db_id="pgvector_single_db",
        )
        self._init_pgvector()

    def _init_pgvector(self):
        with self._pg_database.begin() as connection:
            connection.execute(
                sqlalchemy.text("CREATE EXTENSION IF NOT EXISTS vector")
            )

    def list_collections(self) -> List[CollectionStateInfo]:
        return self._collection_info_cache.list_collections()

    def get_collection(self, collection_id: str) -> Collection:
        return PgvectorCollection(
            pg_database=self._pg_database,
            collection_id=collection_id,
            collection_info_cache=self._collection_info_cache,
        )

    def get_blue_collection(self) -> Optional[Collection]:
        info = self._collection_info_cache.get_blue_collection()
        if info:
            return self.get_collection(info.collection_id)
        return None

    def set_blue_collection(self, collection_id: str) -> None:
        self._collection_info_cache.set_blue_collection(collection_id)

    def create_collection(
        self,
        embedding_model: EmbeddingModel,
        collection_id: Optional[str] = None,
    ) -> Collection:
        if not collection_id:
            collection_id = f"{embedding_model.name}_{embedding_model.id}"

        collection_info = CollectionInfo(
            collection_id=collection_id,
            embedding_model=embedding_model,
        )
        db_model = make_db_model(collection_info)
        db_model.__table__.create(self._pg_database, checkfirst=True)
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
        return self.get_collection(collection_id)

    def delete_collection(self, collection_id: str) -> None:
        col_info = self._collection_info_cache.get_collection(collection_id)
        if not col_info:
            raise CollectionNotFoundError(collection_id)
        if col_info.work_state == CollectionWorkState.BLUE:
            raise DeleteBlueCollectionError()
        db_model = make_db_model(col_info)
        db_model.__table__.drop(self._pg_database, checkfirst=True)
        # TODO: protect from inconsistent state (after crash at this point)
        self._collection_info_cache.delete_collection(collection_id)
