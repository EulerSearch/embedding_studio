from typing import Any, Dict, Generic, List, Optional, Type, TypeVar, Union

from bson import ObjectId
from pydantic import BaseModel
from pymongo.collection import Collection

SchemaInDbType = TypeVar("SchemaInDbType", bound=BaseModel)
CreateSchemaType = TypeVar("CreateSchemaType", bound=BaseModel)
UpdateSchemaType = TypeVar("UpdateSchemaType", bound=BaseModel)


class CRUDBase(Generic[SchemaInDbType, CreateSchemaType, UpdateSchemaType]):
    def __init__(self, collection: Collection, model: Type[SchemaInDbType]):
        self.collection = collection
        self.model = model

    @staticmethod
    def to_object_id(id: Union[str, ObjectId]) -> Optional[ObjectId]:
        if isinstance(id, str):
            if not ObjectId.is_valid(id):
                return None
            return ObjectId(id)
        return id

    def exists(self, id: Union[str, ObjectId]) -> bool:
        obj_id = self.to_object_id(id)
        if not obj_id:
            return False
        if self.collection.count_documents({"_id": obj_id}, limit=1) != 0:
            return True
        return False

    def get(self, id: Union[str, ObjectId]) -> Optional[SchemaInDbType]:
        obj_id = self.to_object_id(id)
        if not obj_id:
            return None

        obj = self.collection.find_one({"_id": obj_id})
        if not obj:
            return None
        return self.model.model_validate(obj)

    def get_all(self, skip: int = 0, limit: int = 100) -> List[SchemaInDbType]:
        objs = list(self.collection.find(skip=skip, limit=limit))
        result: List[SchemaInDbType] = []
        for obj in objs:
            result.append(self.model.model_validate(obj))
        return result

    def create(
        self, schema: CreateSchemaType, return_obj: bool = False
    ) -> Optional[Union[SchemaInDbType, ObjectId]]:
        new_obj = self.collection.insert_one(schema.model_dump())
        if return_obj:
            created_new_obj = self.collection.find_one(
                {"_id": new_obj.inserted_id}
            )
            if created_new_obj:
                return self.model.model_validate(created_new_obj)
            return None
        return new_obj.inserted_id

    def update(
        self,
        obj: SchemaInDbType,
        values: Optional[
            Union[UpdateSchemaType, Dict[str, Any], SchemaInDbType]
        ] = None,
    ) -> Optional[SchemaInDbType]:
        obj_id = self.to_object_id(obj.id)
        if not obj_id:
            return None

        if values:
            if isinstance(values, dict):
                new_values = values
            else:
                new_values = values.model_dump(exclude={"id"})
        else:
            new_values = obj.model_dump(exclude={"id"})

        result = self.collection.update_one(
            {"_id": obj_id},
            {"$set": new_values},
        )
        if result.matched_count > 0:
            return self.get(id=obj.id)
        return None

    def remove(self, id: Union[str, ObjectId]) -> bool:
        obj_id = self.to_object_id(id)
        if not obj_id:
            return False

        deleted_obj = self.collection.delete_one({"_id": obj_id})
        if deleted_obj.deleted_count != 1:
            return False
        return True
