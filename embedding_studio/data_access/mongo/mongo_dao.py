import os
from collections.abc import Iterable
from typing import Any, Dict, Generic, List, Optional, Set, Type, TypeVar

import pymongo
from bson import ObjectId

ModelT = TypeVar("ModelT")


class MongoDao(Generic[ModelT]):
    """
    A generic Data Access Object for MongoDB collections.

    This class provides a mapping layer between MongoDB collections and Pydantic models,
    handling conversion between BSON documents and model instances.

    :param collection: MongoDB collection to operate on
    :param model: Pydantic model class to use for object mapping
    :param model_id: Field name of the model's ID
    :param model_mongo_id: Field name to map MongoDB's _id to in the model (defaults to None)
    :param additional_indexes: List of additional indexes to create on the collection
    """

    def __init__(
        self,
        collection: pymongo.collection.Collection,
        model: Type[ModelT],
        model_id: str,
        model_mongo_id: Optional[str] = None,
        additional_indexes: Optional[List[Dict[str, Any]]] = None,
    ):
        self.collection = collection
        self.model = model
        self.model_id = model_id
        self.model_mongo_id = model_mongo_id
        self._init_indexes(additional_indexes)

    def _init_indexes(
        self, additional_indexes: Optional[List[Dict[str, Any]]] = None
    ):
        """
        Initialize indexes for the MongoDB collection.

        Creates a unique index on model_id if it differs from model_mongo_id,
        and creates any additional specified indexes.

        :param additional_indexes: List of additional indexes to create on the collection
        """
        if self.model_mongo_id != self.model_id:
            self.collection.create_index(
                self.model_id, unique=True, background=True
            )
        # TODO: remove this when we have a better way to run unit tests
        if os.getenv("ES_UNIT_TESTS") != "1":
            if additional_indexes:
                for index in additional_indexes:
                    self.collection.create_index(**index, background=True)

    def bson_to_model(self, bson: Any) -> ModelT:
        """
        Convert a BSON document to a model instance.

        :param bson: BSON document from MongoDB
        :return: Instance of the model class
        """
        bson = dict(bson)
        if self.model_mongo_id:
            bson[self.model_mongo_id] = str(bson.pop("_id"))
        return self.model.model_validate(bson)

    def bson_to_model_opt(self, bson: Any) -> Optional[ModelT]:
        """
        Convert a BSON document to a model instance, handling None values.

        :param bson: BSON document from MongoDB, or None
        :return: Instance of the model class, or None if input is None
        """
        if bson:
            return self.bson_to_model(bson)
        return None

    def bsons_to_models(self, bsons: Iterable[Any]) -> List[ModelT]:
        """
        Convert multiple BSON documents to model instances.

        :param bsons: Iterable of BSON documents
        :return: List of model instances
        """
        return [self.bson_to_model(bson) for bson in bsons]

    def model_to_bson(
        self, obj: ModelT, set_id: bool = False, **model_dump_kwargs
    ) -> Dict[str, Any]:
        """
        Convert a model instance to a BSON document.

        :param obj: Model instance to convert
        :param set_id: Whether to set the MongoDB _id field
        :param model_dump_kwargs: Additional arguments to pass to model_dump
        :return: BSON document
        """
        result = obj.model_dump(
            mode="json", exclude_none=True, **model_dump_kwargs
        )
        if self.model_mongo_id and self.model_mongo_id in result:
            mongo_id = result.pop(self.model_mongo_id)
            if set_id:
                result["_id"] = ObjectId(mongo_id)
        return result

    def models_to_bsons(self, objs: Iterable[ModelT]) -> List[Dict[str, Any]]:
        """
        Convert multiple model instances to BSON documents.

        :param objs: Iterable of model instances
        :return: List of BSON documents
        """
        return [self.model_to_bson(obj) for obj in objs]

    def get_schema_properties(self) -> Set[str]:
        """
        Get all property names from the model's schema.

        :return: Set of property names
        """
        schema = self.model.schema()
        return set(schema["properties"].keys())

    def get_model_projection(self) -> Dict[str, bool]:
        """
        Create a MongoDB projection for the model's properties.

        This projection ensures only the fields defined in the model schema are returned,
        and handles the mapping between _id and model_mongo_id if necessary.

        :return: MongoDB projection document
        """
        props = self.get_schema_properties()
        projection = {prop: True for prop in props}
        if self.model_mongo_id:
            projection.pop(self.model_mongo_id)
            projection["_id"] = True
        else:
            projection["_id"] = False
        return projection

    def model_id_to_db_id(self, obj_id: Any) -> (str, Any):
        """
        Convert a model ID to a database ID.

        If model_id and model_mongo_id are the same, converts to MongoDB ObjectId.

        :param obj_id: Model ID value
        :return: Tuple of (ID field name, ID value)
        """
        if self.model_id == self.model_mongo_id:
            return "_id", ObjectId(obj_id)
        return self.model_id, obj_id

    def get_db_id(self, obj: ModelT) -> (str, Any):
        """
        Get the database ID from a model instance.

        :param obj: Model instance
        :return: Tuple of (ID field name, ID value)
        """
        value = getattr(obj, self.model_id)
        return self.model_id_to_db_id(value)

    def find_one(
        self, obj_id: Optional[Any] = None, **kwargs
    ) -> Optional[ModelT]:
        """
        Find a single document by ID or custom filter.

        :param obj_id: ID value to search by
        :param kwargs: Additional arguments to pass to MongoDB's find_one
        :return: Model instance if found, None otherwise
        """
        if obj_id is not None:
            id_name, id_value = self.model_id_to_db_id(obj_id)
            if not kwargs.get("filter"):
                kwargs["filter"] = dict()

            kwargs["filter"].update({id_name: id_value})
        projection = self.get_model_projection()
        result_bson = self.collection.find_one(**kwargs, projection=projection)
        return self.bson_to_model_opt(result_bson)

    def find(
        self, sort_args: Optional[Any] = None, **find_kwargs
    ) -> List[ModelT]:
        """
        Find documents matching a filter.

        :param sort_args: Arguments to pass to MongoDB's sort
        :param find_kwargs: Additional arguments like 'filter' and 'limit' to pass to MongoDB's find
        :return: List of model instances
        """
        projection = self.get_model_projection()
        cursor = self.collection.find(
            find_kwargs.get("filter", dict()), projection=projection
        )
        if sort_args:
            cursor = cursor.sort(*sort_args)

        if "limit" in find_kwargs:
            cursor = cursor.limit(find_kwargs["limit"])

        bsons = [bson for bson in cursor]
        return self.bsons_to_models(bsons)

    def insert_one(
        self, obj: ModelT, **kwargs
    ) -> pymongo.results.InsertOneResult:
        """
        Insert a single document.

        :param obj: Model instance to insert
        :param kwargs: Additional arguments to pass to MongoDB's insert_one
        :return: MongoDB InsertOneResult
        """
        return self.collection.insert_one(
            document=self.model_to_bson(obj), **kwargs
        )

    def insert_many(
        self, objs: Iterable[ModelT], **kwargs
    ) -> pymongo.results.InsertManyResult:
        """
        Insert multiple documents.

        :param objs: Iterable of model instances to insert
        :param kwargs: Additional arguments to pass to MongoDB's insert_many
        :return: MongoDB InsertManyResult
        """
        return self.collection.insert_many(
            documents=self.models_to_bsons(objs), **kwargs
        )

    def update_one(
        self, obj: Optional[ModelT] = None, **kwargs
    ) -> pymongo.results.UpdateResult:
        """
        Update a single document.

        If obj is provided, updates the document with matching ID with obj's data.
        Otherwise, uses the provided filter and update parameters.

        :param obj: Model instance containing updated data
        :param kwargs: Additional arguments to pass to MongoDB's update_one
        :return: MongoDB UpdateResult
        """
        if obj is not None:
            if "update" not in kwargs:
                kwargs["update"] = {"$set": self.model_to_bson(obj)}
            if "filter" not in kwargs:
                db_id, value = self.get_db_id(obj)
                kwargs["filter"] = {db_id: value}

        return self.collection.update_one(**kwargs)

    def upsert_one(
        self, obj: Optional[ModelT] = None, **kwargs
    ) -> pymongo.results.UpdateResult:
        """
        Update a single document or insert it if it doesn't exist.

        :param obj: Model instance containing the data
        :param kwargs: Additional arguments to pass to MongoDB's update_one
        :return: MongoDB UpdateResult
        """
        return self.update_one(obj, **kwargs, upsert=True)

    def find_one_and_update(
        self, obj_id: Optional[Any] = None, **kwargs
    ) -> Optional[ModelT]:
        """
        Find a single document and update it.

        :param obj_id: ID value to search by
        :param kwargs: Additional arguments to pass to MongoDB's find_one_and_update
        :return: Updated model instance if found, None otherwise
        """
        if obj_id is not None:
            id_name, id_value = self.model_id_to_db_id(obj_id)
            kwargs["filter"] = {id_name: id_value}
        projection = self.get_model_projection()
        result_bson = self.collection.find_one_and_update(
            **kwargs, projection=projection
        )
        return self.bson_to_model_opt(result_bson)

    def delete_one(
        self, obj_id: Optional[Any] = None, **kwargs
    ) -> pymongo.results.DeleteResult:
        """
        Delete a single document.

        :param obj_id: ID value to delete
        :param kwargs: Additional arguments to pass to MongoDB's delete_one
        :return: MongoDB DeleteResult
        """
        if obj_id is not None:
            id_name, id_value = self.model_id_to_db_id(obj_id)
            kwargs["filter"] = {id_name: id_value}
        return self.collection.delete_one(**kwargs)

    def find_one_and_delete(
        self, obj_id: Optional[Any] = None, **kwargs
    ) -> Optional[ModelT]:
        """
        Find a single document and delete it.

        :param obj_id: ID value to search by
        :param kwargs: Additional arguments to pass to MongoDB's find_one_and_delete
        :return: Deleted model instance if found, None otherwise
        """
        if obj_id is not None:
            id_name, id_value = self.model_id_to_db_id(obj_id)
            kwargs["filter"] = {id_name: id_value}
        projection = self.get_model_projection()
        result_bson = self.collection.find_one_and_delete(
            **kwargs, projection=projection
        )
        return self.bson_to_model_opt(result_bson)
