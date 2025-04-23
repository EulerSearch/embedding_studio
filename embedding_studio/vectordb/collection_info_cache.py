import datetime
import logging
from typing import List, Optional

import pymongo
from pydantic import BaseModel, Field

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
    """
    A cache for collection information and states.

    This class manages collection metadata and state information in MongoDB,
    providing methods to add, update, retrieve, and manage collections and query collections.

    :param mongo_database: MongoDB database instance
    :param db_id: Database identifier
    """

    class BlueCollectionId(BaseModel):
        """
        Model representing the current 'blue' (primary active) collection IDs.

        :param db_id: Database identifier
        :param collection_id: ID of the blue collection
        :param query_collection_id: ID of the blue query collection
        """

        db_id: str
        collection_id: str
        query_collection_id: Optional[str] = Field(default=None)

    class CollectionInfoDb(CollectionInfo):
        """
        Extended CollectionInfo model with database-specific fields.

        :param db_id: Database identifier
        :param index_created: Whether an index has been created for this collection
        :param contains_queries: Whether this collection contains queries
        :param created_at: Creation timestamp
        """

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
        """
        Refresh the in-memory cache of collections from the database.

        This method clears the current cache and reloads all collection information
        from MongoDB. It also identifies and marks the current blue collections.
        During this process, it:
        1. Clears all in-memory collections
        2. Fetches all collections for this database from MongoDB
        3. Fetches the blue collection identifiers
        4. Sorts collections into regular and query collections
        5. Identifies and marks the blue collections

        :return: None
        """
        self._collections = []
        self._query_collections = []

        self._blue_collection = None

        db_collections = self._collection_info_dao.find(
            filter={self._DB_ID: self._db_id}
        )
        blue_collection_info = self._blue_collection_id_dao.find_one(
            filter={self._DB_ID: self._db_id}
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
        """
        Get a list of all regular collections.

        :return: List of collection state information objects
        """
        return self._collections

    def list_query_collections(self) -> List[CollectionStateInfo]:
        """
        Get a list of all query collections.

        :return: List of query collection state information objects
        """
        return self._query_collections

    def get_collection(
        self, collection_id: str
    ) -> Optional[CollectionStateInfo]:
        """
        Find a collection by its ID.

        :param collection_id: ID of the collection to find
        :return: Collection state information or None if not found
        """
        for collection in self._collections:
            if str(collection.collection_id) == str(collection_id):
                return collection

        for collection in self._query_collections:
            if str(collection.collection_id) == str(collection_id):
                return collection

        return None

    def get_blue_collection(self) -> Optional[CollectionStateInfo]:
        """
        Get the current blue (primary active) collection.

        :return: Blue collection state information or None if not set
        """
        return self._blue_collection

    def get_blue_query_collection(self) -> Optional[CollectionStateInfo]:
        """
        Get the current blue (primary active) query collection.

        :return: Blue query collection state information or None if not set
        """
        return self._blue_query_collection

    def set_blue_collection(
        self, collection_id: str, query_collection_id: str
    ) -> None:
        """
        Set the blue (primary active) collection and query collection.

        :param collection_id: ID of the collection to set as blue
        :param query_collection_id: ID of the query collection to set as blue
        :return: None
        :raises CollectionNotFoundError: If either collection is not found
        """
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
        """
        Update the index creation state for a collection.

        :param collection_id: ID of the collection
        :param created: Whether the index has been created
        :return: None
        """
        self._collection_info_dao.update_one(
            filter={self._COLLECTION_ID: collection_id},
            update={"$set": {self._INDEX_CREATED: created}},
        )
        self.invalidate_cache()

    def add_collection(
        self, collection_info: CollectionInfo
    ) -> CollectionStateInfo:
        """
        Add a new regular collection to the database.

        This method:
        1. Creates a database model from the collection info
        2. Sets required fields (created_at, db_id, index_created, contains_queries=False)
        3. Inserts the collection into MongoDB
        4. Refreshes the cache to include the new collection
        5. Returns the newly created collection state

        Duplicates are handled gracefully with a warning log.

        :param collection_info: Information about the collection to add
        :return: State information for the added collection
        """
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

    def update_collection(
        self, collection_info: CollectionInfo
    ) -> CollectionStateInfo:
        """
        Update an existing regular collection in the database.

        This method:
        1. Prepares an update payload from the collection info
        2. Adds an updated_at timestamp to track the last modification
        3. Updates the collection document in MongoDB
        4. Refreshes the cache to reflect the changes
        5. Returns the updated collection state

        If the collection is not found, a warning is logged.

        :param collection_info: Updated information for the collection
        :return: State information for the updated collection
        """
        # Prepare the update payload from the collection_info.
        update_data = collection_info.model_dump()
        update_data["updated_at"] = current_time()  # Add an updated timestamp

        # Use update_one to modify the document matching the collection_id.
        result = self._collection_info_dao.update_one(
            filter={"collection_id": collection_info.collection_id},
            update={"$set": update_data},
        )

        if result.matched_count == 0:
            logger.warning(
                f"collection {collection_info.collection_id} not found for update"
            )
        else:
            logger.info(
                f"collection {collection_info.collection_id} updated successfully"
            )

        self.invalidate_cache()
        collection = self.get_collection(collection_info.collection_id)
        return collection

    def add_query_collection(
        self, collection_info: CollectionInfo
    ) -> CollectionStateInfo:
        """
        Add a new query collection to the database.

        This method:
        1. Creates a database model from the collection info
        2. Sets required fields (created_at, db_id, index_created, contains_queries=True)
        3. Inserts the collection into MongoDB, marking it specifically as a query collection
        4. Refreshes the cache to include the new query collection
        5. Returns the newly created query collection state

        The main difference from add_collection is that contains_queries is set to True,
        which ensures this collection is tracked as a query collection.

        :param collection_info: Information about the query collection to add
        :return: State information for the added query collection
        """
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

    def update_query_collection(
        self, collection_info: CollectionInfo
    ) -> CollectionStateInfo:
        """
        Update an existing query collection in the database.

        This method:
        1. Prepares an update payload from the collection info
        2. Adds an updated_at timestamp to track the last modification
        3. Ensures the collection remains marked as a query collection (contains_queries=True)
        4. Updates the query collection document in MongoDB
        5. Refreshes the cache to reflect the changes
        6. Returns the updated query collection state

        The key difference from update_collection is that this method enforces the
        contains_queries=True flag to maintain the collection's status as a query collection.

        :param collection_info: Updated information for the query collection
        :return: State information for the updated query collection
        """
        # Prepare update payload from the collection_info.
        update_data = collection_info.model_dump()
        update_data["updated_at"] = current_time()  # Add an updated timestamp
        # Ensure the collection remains marked as containing queries.
        update_data["contains_queries"] = True

        # Use update_one to update the document matching the collection_id.
        result = self._collection_info_dao.update_one(
            filter={"collection_id": collection_info.collection_id},
            update={"$set": update_data},
        )

        if result.matched_count == 0:
            logger.warning(
                f"collection {collection_info.collection_id} not found for update"
            )
        else:
            logger.info(
                f"collection {collection_info.collection_id} updated successfully"
            )

        self.invalidate_cache()
        collection = self.get_collection(collection_info.collection_id)
        return collection

    def delete_collection(self, collection_id: str) -> None:
        """
        Delete a collection from the database.

        :param collection_id: ID of the collection to delete
        :return: None
        """
        self._collection_info_dao.delete_one(collection_id)
        self.invalidate_cache()
