from typing import Optional, Type, Union

from bson import ObjectId
from pymongo.collection import Collection

from embedding_studio.data_access.mongo.crud_base import (
    CreateSchemaType,
    CRUDBase,
    SchemaInDbType,
    UpdateSchemaType,
)


class CRUDModelTransferTasks(
    CRUDBase[SchemaInDbType, CreateSchemaType, UpdateSchemaType]
):
    _EMBEDDING_MODEL_ID: str = "embedding_model_id"
    _DST_EMBEDDING_MODEL_ID: str = "dst_embedding_model_id"

    def __init__(
        self,
        collection: Collection,
        model: Type[SchemaInDbType],
    ):
        super(CRUDModelTransferTasks, self).__init__(
            collection,
            model,
            [
                CRUDModelTransferTasks._EMBEDDING_MODEL_ID,
                CRUDModelTransferTasks._DST_EMBEDDING_MODEL_ID,
            ],
        )

    def get_by_model_id(
        self, embedding_model_id: Union[str, ObjectId]
    ) -> Optional[SchemaInDbType]:
        """Get an object by embedding model ID.

        :param embedding_model_id: ID as string or ObjectId.
        :return: Retrieved object or None if not found.
        """

        obj = self.collection.find_one(
            {self._EMBEDDING_MODEL_ID: embedding_model_id}
        )
        if not obj:
            return None

        return self.model.model_validate(obj)

    def get_by_dst_model_id(
        self, embedding_model_id: Union[str, ObjectId]
    ) -> Optional[SchemaInDbType]:
        """Get an object by destination embedding model ID.

        :param embedding_model_id: ID as string or ObjectId.
        :return: Retrieved object or None if not found.
        """

        obj = self.collection.find_one(
            {self._DST_EMBEDDING_MODEL_ID: embedding_model_id}
        )
        if not obj:
            return None

        return self.model.model_validate(obj)
