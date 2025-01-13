import datetime
import logging
from typing import List, Optional

import pymongo
from pydantic import BaseModel

from embedding_studio.data_access.mongo.mongo_dao import MongoDao
from embedding_studio.models.embeddings.collections import (
    CollectionInfo,
    CollectionStateInfo,
    CollectionWorkState,
)
from embedding_studio.utils.datetime_utils import current_time
from embedding_studio.vectordb.exceptions import CollectionNotFoundError

logger = logging.getLogger(__name__)


class CollectionInfoCache:
    class BlueCollectionId(BaseModel):
        db_id: str
        collection_id: str
        query_collection_id: str

    class CollectionInfoDb(CollectionInfo):
        db_id: str
        index_created: bool
        contains_queries: bool
        created_at: datetime.datetime

    _MONGO_COLLECTION_INFO = "vectordb_collection_info"
    _MONGO_COLLECTION_BLUE_ID = "vectordb_blue_collection_id"

    _DB_ID = "db_id"
    _COLLECTION_ID = "collection_id"
    _INDEX_CREATED = "index_created"

    def __init__(self, mongo_database: pymongo.database.Database, db_id: str):
        self._db_id = db_id
        self._collection_info_dao = MongoDao[self.CollectionInfoDb](
            collection=mongo_database[self._MONGO_COLLECTION_INFO],
            model=self.CollectionInfoDb,
            model_id=self._COLLECTION_ID,
            additional_indexes=[dict(keys=self._DB_ID)],
        )
        self._blue_collection_id_dao = MongoDao[self.BlueCollectionId](
            collection=mongo_database[self._MONGO_COLLECTION_BLUE_ID],
            model=self.BlueCollectionId,
            model_id=self._DB_ID,
        )

        self._collections: List[CollectionStateInfo] = []
        self._query_collections: List[CollectionStateInfo] = []

        self._blue_collection: Optional[CollectionStateInfo] = None
        self._blue_query_collection: Optional[CollectionStateInfo] = None

        self.invalidate_cache()

    def invalidate_cache(self):
        self._collections = []
        self._blue_collection = None
        db_collections = self._collection_info_dao.find(
            filter={self._DB_ID: self._db_id}
        )
        blue_collection_info = self._blue_collection_id_dao.find_one(
            self._db_id
        )

        blue_collection_id = None
        blue_query_collection_id = None
        if blue_collection_info:
            blue_collection_id = blue_collection_info.collection_id
            blue_query_collection_id = blue_collection_info.query_collection_id

        for db_collection in db_collections:
            collection_state_info = CollectionStateInfo(
                **db_collection.model_dump(exclude={"db_id"}),
                work_state=CollectionWorkState.GREEN,
            )
            if db_collection.contains_queries:
                self._query_collections.append(collection_state_info)
            else:
                self._collections.append(collection_state_info)
            if (
                blue_collection_id
                and db_collection.collection_id == blue_collection_id
            ):
                self._blue_collection = collection_state_info

            elif (
                blue_query_collection_id
                and db_collection.collection_id == blue_query_collection_id
            ):
                self._blue_query_collection = collection_state_info

    def list_collections(self) -> List[CollectionStateInfo]:
        return self._collections

    def list_query_collections(self) -> List[CollectionStateInfo]:
        return self._query_collections

    def get_collection(
        self, collection_id: str
    ) -> Optional[CollectionStateInfo]:
        for collection in self._collections:
            if str(collection.collection_id) == str(collection_id):
                return collection

        for collection in self._query_collections:
            if str(collection.collection_id) == str(collection_id):
                return collection

        return None

    def get_blue_collection(self) -> Optional[CollectionStateInfo]:
        return self._blue_collection

    def get_blue_query_collection(self) -> Optional[CollectionStateInfo]:
        return self._blue_query_collection

    def set_blue_collection(
        self, collection_id: str, query_collection_id: str
    ) -> None:
        self.invalidate_cache()

        info = self.get_collection(collection_id)
        if not info:
            raise CollectionNotFoundError(collection_id)

        query_info = self.get_collection(query_collection_id)
        if not query_info:
            raise CollectionNotFoundError(query_collection_id)

        blue_id = self.BlueCollectionId(
            db_id=self._db_id,
            collection_id=collection_id,
            query_collection_id=query_collection_id,
        )

        self._blue_collection_id_dao.upsert_one(blue_id)

        self.invalidate_cache()

    def set_index_state(self, collection_id: str, created: bool):
        self._collection_info_dao.update_one(
            filter={self._COLLECTION_ID: collection_id},
            update={"$set": {self._INDEX_CREATED: created}},
        )
        self.invalidate_cache()

    def add_collection(
        self, collection_info: CollectionInfo
    ) -> CollectionStateInfo:
        collection_info_db = self.CollectionInfoDb(
            **collection_info.model_dump(),
            created_at=current_time(),
            db_id=self._db_id,
            index_created=False,
            contains_queries=False,
        )
        try:
            self._collection_info_dao.insert_one(collection_info_db)
        except pymongo.errors.DuplicateKeyError:
            logger.warning(
                f"collection {collection_info.collection_id} already exists"
            )
        self.invalidate_cache()
        collection = self.get_collection(collection_info.collection_id)
        return collection

    def add_query_collection(
        self, collection_info: CollectionInfo
    ) -> CollectionStateInfo:
        collection_info_db = self.CollectionInfoDb(
            **collection_info.model_dump(),
            created_at=current_time(),
            db_id=self._db_id,
            index_created=False,
            contains_queries=True,
        )
        try:
            self._collection_info_dao.insert_one(collection_info_db)
        except pymongo.errors.DuplicateKeyError:
            logger.warning(
                f"collection {collection_info.collection_id} already exists"
            )
        self.invalidate_cache()
        collection = self.get_collection(collection_info.collection_id)
        return collection

    def delete_collection(self, collection_id: str) -> None:
        self._collection_info_dao.delete_one(collection_id)
        self.invalidate_cache()
