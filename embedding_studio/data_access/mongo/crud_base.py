import uuid
from datetime import datetime
from typing import (
    Any,
    Dict,
    Generic,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)

from bson import ObjectId
from pydantic import BaseModel
from pymongo.collection import Collection

SchemaInDbType = TypeVar("SchemaInDbType", bound=BaseModel)
CreateSchemaType = TypeVar("CreateSchemaType", bound=BaseModel)
UpdateSchemaType = TypeVar("UpdateSchemaType", bound=BaseModel)


class CRUDBase(Generic[SchemaInDbType, CreateSchemaType, UpdateSchemaType]):
    _MONGODB_ID: str = "_id"
    _MONGODB_SET: str = "$set"
    _IDEMPOTENCY_KEY: str = "idempotency_key"
    _UPDATED_AT: str = "updated_at"

    def __init__(
        self,
        collection: Collection,
        model: Type[SchemaInDbType],
        indexes: List[Union[str, Tuple[str, ...]]] = None,
    ):
        """Initialize the CRUDBase class.

        :param collection: MongoDB's collection.
        :param model: Pydantic model type.
        :param indexes: List of indexes to be created on the collection.
        """
        self.collection = collection
        self.model = model

        if indexes:
            self.collection.create_index(indexes)

    @staticmethod
    def to_object_id(id: Union[str, ObjectId]) -> Optional[ObjectId]:
        """Convert a string or ObjectId to ObjectId.

        :param id: ID as string or ObjectId.
        :return: Converted ObjectId or None.
        """
        if isinstance(id, str):
            if not ObjectId.is_valid(id):
                return None
            return ObjectId(id)
        return id

    def exists(self, id: Union[str, ObjectId]) -> bool:
        """Check if an object with the specified ID exists.

        :param id: ID as string or ObjectId.
        :return: True if the object exists, False otherwise.
        """
        obj_id = self.to_object_id(id)
        if not obj_id:
            return False
        if (
            self.collection.count_documents(
                {self._MONGODB_ID: obj_id}, limit=1
            )
            != 0
        ):
            return True
        return False

    def get(self, id: Union[str, ObjectId]) -> Optional[SchemaInDbType]:
        """Get an object by ID.

        :param id: ID as string or ObjectId.
        :return: Retrieved object or None if not found.
        """
        obj_id = self.to_object_id(id)
        if not obj_id:
            return None

        obj = self.collection.find_one({self._MONGODB_ID: obj_id})
        if not obj:
            return None
        return self.model.model_validate(obj)

    def get_by_filter(
        self, filter: Dict[str, Any], skip: int = 0, limit: int = 100
    ) -> List[SchemaInDbType]:
        """Get a list of objects based on a filter.

        :param filter: MongoDB filter.
        :param skip: Number of objects to skip.
        :param limit: Maximum number of objects to retrieve.
        :return: List of retrieved objects.
        """
        objs = self.collection.find(filter, skip=skip, limit=limit)
        result: List[SchemaInDbType] = [
            self.model.model_validate(obj) for obj in objs
        ]
        return result

    def get_all(self, skip: int = 0, limit: int = 100) -> List[SchemaInDbType]:
        """Get all objects.

        :param skip: Number of objects to skip.
        :param limit: Maximum number of objects to retrieve.
        :return: List of retrieved objects.
        """
        objs = list(self.collection.find(skip=skip, limit=limit))
        result: List[SchemaInDbType] = [
            self.model.model_validate(obj) for obj in objs
        ]
        return result

    def get_by_idempotency_key(
        self, idempotency_key: uuid.UUID
    ) -> Optional[SchemaInDbType]:
        """Get an object by idempotency key.

        :param idempotency_key: Idempotency key as UUID.
        :return: Retrieved object or None if not found.
        """
        obj = self.collection.find_one(
            {self._IDEMPOTENCY_KEY: idempotency_key}
        )
        if not obj:
            return None
        return self.model.model_validate(obj)

    def create(
        self, schema: CreateSchemaType, return_obj: bool = False
    ) -> Optional[Union[SchemaInDbType, ObjectId]]:
        """Create a new object.

        :param schema: Pydantic model instance for creation.
        :param return_obj: Flag to indicate if the created object should be
            returned.
        :return: Created object or ID.
        """
        new_obj = self.collection.insert_one(schema.model_dump())
        if return_obj:
            created_new_obj = self.collection.find_one(
                {self._MONGODB_ID: new_obj.inserted_id}
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
        """Update an existing object.

        :param obj: Object to be updated.
        :param values: Values to update.
        :return: Updated object or None if not found.
        """
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

        if self._UPDATED_AT in new_values:
            new_values[self._UPDATED_AT] = datetime.utcnow()

        result = self.collection.update_one(
            {self._MONGODB_ID: obj_id},
            {self._MONGODB_SET: new_values},
        )
        if result.matched_count > 0:
            return self.get(id=obj.id)
        return None

    def remove(self, id: Union[str, ObjectId]) -> bool:
        """Remove an object by ID.

        :param id: ID as string or ObjectId.
        :return: True if the object is removed successfully, False otherwise.
        """
        obj_id = self.to_object_id(id)
        if not obj_id:
            return False

        deleted_obj = self.collection.delete_one({self._MONGODB_ID: obj_id})
        if deleted_obj.deleted_count != 1:
            return False
        return True
