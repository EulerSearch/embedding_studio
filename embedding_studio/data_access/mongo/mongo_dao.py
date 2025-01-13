import os
from collections.abc import Iterable
from typing import Any, Dict, Generic, List, Optional, Set, Type, TypeVar

import pymongo
from bson import ObjectId

ModelT = TypeVar("ModelT")


class MongoDao(Generic[ModelT]):
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
        bson = dict(bson)
        if self.model_mongo_id:
            bson[self.model_mongo_id] = str(bson.pop("_id"))
        return self.model.model_validate(bson)

    def bson_to_model_opt(self, bson: Any) -> Optional[ModelT]:
        if bson:
            return self.bson_to_model(bson)
        return None

    def bsons_to_models(self, bsons: Iterable[Any]) -> List[ModelT]:
        return [self.bson_to_model(bson) for bson in bsons]

    def model_to_bson(
        self, obj: ModelT, set_id: bool = False, **model_dump_kwargs
    ) -> Dict[str, Any]:
        result = obj.model_dump(
            mode="json", exclude_none=True, **model_dump_kwargs
        )
        if self.model_mongo_id and self.model_mongo_id in result:
            mongo_id = result.pop(self.model_mongo_id)
            if set_id:
                result["_id"] = ObjectId(mongo_id)
        return result

    def models_to_bsons(self, objs: Iterable[ModelT]) -> List[Dict[str, Any]]:
        return [self.model_to_bson(obj) for obj in objs]

    def get_schema_properties(self) -> Set[str]:
        schema = self.model.schema()
        return set(schema["properties"].keys())

    def get_model_projection(self) -> Dict[str, bool]:
        props = self.get_schema_properties()
        projection = {prop: True for prop in props}
        if self.model_mongo_id:
            projection.pop(self.model_mongo_id)
            projection["_id"] = True
        else:
            projection["_id"] = False
        return projection

    def model_id_to_db_id(self, obj_id: Any) -> (str, Any):
        if self.model_id == self.model_mongo_id:
            return "_id", ObjectId(obj_id)
        return self.model_id, obj_id

    def get_db_id(self, obj: ModelT) -> (str, Any):
        value = getattr(obj, self.model_id)
        return self.model_id_to_db_id(value)

    def find_one(
        self, obj_id: Optional[Any] = None, **kwargs
    ) -> Optional[ModelT]:
        if obj_id is not None:
            id_name, id_value = self.model_id_to_db_id(obj_id)
            kwargs["filter"] = kwargs.get("filter", dict()).update(
                {id_name: id_value}
            )
        projection = self.get_model_projection()
        result_bson = self.collection.find_one(**kwargs, projection=projection)
        return self.bson_to_model_opt(result_bson)

    def find(
        self, sort_args: Optional[Any] = None, **find_kwargs
    ) -> List[ModelT]:
        projection = self.get_model_projection()
        cursor = self.collection.find(**find_kwargs, projection=projection)
        if sort_args:
            cursor = cursor.sort(*sort_args)
        return self.bsons_to_models(cursor)

    def insert_one(
        self, obj: ModelT, **kwargs
    ) -> pymongo.results.InsertOneResult:
        return self.collection.insert_one(
            document=self.model_to_bson(obj), **kwargs
        )

    def insert_many(
        self, objs: Iterable[ModelT], **kwargs
    ) -> pymongo.results.InsertManyResult:
        return self.collection.insert_many(
            documents=self.models_to_bsons(objs), **kwargs
        )

    def update_one(
        self, obj: Optional[ModelT] = None, **kwargs
    ) -> pymongo.results.UpdateResult:
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
        return self.update_one(obj, **kwargs, upsert=True)

    def find_one_and_update(
        self, obj_id: Optional[Any] = None, **kwargs
    ) -> Optional[ModelT]:
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
        if obj_id is not None:
            id_name, id_value = self.model_id_to_db_id(obj_id)
            kwargs["filter"] = {id_name: id_value}
        return self.collection.delete_one(**kwargs)

    def find_one_and_delete(
        self, obj_id: Optional[Any] = None, **kwargs
    ) -> Optional[ModelT]:
        if obj_id is not None:
            id_name, id_value = self.model_id_to_db_id(obj_id)
            kwargs["filter"] = {id_name: id_value}
        projection = self.get_model_projection()
        result_bson = self.collection.find_one_and_delete(
            **kwargs, projection=projection
        )
        return self.bson_to_model_opt(result_bson)
