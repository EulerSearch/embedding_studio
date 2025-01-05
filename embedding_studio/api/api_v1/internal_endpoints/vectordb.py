import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, status

from embedding_studio.api.api_v1.internal_schemas.vectrordb import (
    CreateCollectionRequest,
    CreateIndexRequest,
    DeleteCollectionRequest,
    DeleteObjectRequest,
    FindObjectsByIdsRequest,
    FindSimilarObjectsRequest,
    GetCollectionInfoRequest,
    InsertObjectsRequest,
    ListCollectionsResponse,
    SetBlueCollectionRequest,
    UpsertObjectsRequest,
)
from embedding_studio.context.app_context import context
from embedding_studio.core.config import settings
from embedding_studio.core.plugin import PluginManager
from embedding_studio.models.embeddings.models import EmbeddingModelInfo
from embedding_studio.vectordb.collection import Collection
from embedding_studio.vectordb.exceptions import (
    CollectionNotFoundError,
    DeleteBlueCollectionError,
)

logger = logging.getLogger(__name__)

router = APIRouter()

plugin_manager = PluginManager()
# Initialize and discover plugins
plugin_manager.discover_plugins(directory=settings.ES_PLUGINS_PATH)


def get_collection(
    model: Optional[EmbeddingModelInfo] = None,
) -> Optional[Collection]:
    if not model:
        return context.vectordb.get_blue_collection()

    return context.vectordb.get_collection(model)


@router.post(
    "/collections/create",
    status_code=status.HTTP_200_OK,
)
def create_collection(body: CreateCollectionRequest):
    plugin = plugin_manager.get_plugin(body.embedding_model.name)
    search_index_info = plugin.get_search_index_info()
    collection = context.vectordb.create_collection(
        body.embedding_model, search_index_info
    )
    info = collection.get_state_info()
    logger.debug(f"collection created: {info.model_dump()}")
    return info


@router.post(
    "/collections/create-index",
    status_code=status.HTTP_200_OK,
)
def create_index(body: CreateIndexRequest):
    collection = get_collection(body.embedding_model)
    collection.create_index()


@router.post(
    "/collections/delete",
    status_code=status.HTTP_200_OK,
)
def delete_collection(body: DeleteCollectionRequest):
    try:
        collection = get_collection(body.embedding_model)

        info = collection.get_state_info()
        logger.debug(f"Delete collection: {info}")
    except Exception:
        pass

    try:
        context.vectordb.delete_collection(body.embedding_model)
    except CollectionNotFoundError as err:
        logger.warning(f"Can't find collection wile deleting: {err}")
        return
    except DeleteBlueCollectionError as err:
        raise HTTPException(
            status_code=status.HTTP_406_NOT_ACCEPTABLE, detail=f"{err}"
        )

    try:
        collection = get_collection(body.embedding_model)
        info = collection.get_state_info()
        logger.error(f"Found deleted collection: {info}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Collection not deleted",
        )
    except CollectionNotFoundError as err:
        logger.debug(f"Collection successfully deleted: {err}")


@router.get(
    "/collections/list",
    status_code=status.HTTP_200_OK,
)
def list_collections():
    collection_infos = context.vectordb.list_collections()
    logger.debug(f"Found collections: {collection_infos}")
    return ListCollectionsResponse(collections=collection_infos)


@router.get(
    "/collections/get-info",
    status_code=status.HTTP_200_OK,
)
def get_collection_info(body: GetCollectionInfoRequest):
    try:
        collection = get_collection(body.embedding_model)
        info = collection.get_state_info()
        logger.debug(f"Found collection: {info}")
        return info
    except CollectionNotFoundError:
        msg = f"Collection with id={body.embedding_model.full_name} not found"
        logger.debug(msg)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=msg,
        )


@router.post(
    "/collections/set-blue",
    status_code=status.HTTP_200_OK,
)
def set_blue_collection(body: SetBlueCollectionRequest):
    try:
        context.vectordb.set_blue_collection(body.embedding_model)
    except CollectionNotFoundError as err:
        logger.debug(f"Collection not found: {err}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Collection not found",
        )
    collection = context.vectordb.get_blue_collection()
    info = collection.get_state_info()
    logger.debug(f"Blue collection set: {info}")
    if info.collection_id != body.embedding_model.full_name:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Something went wrong collection_id {info.collection_id} != "
            f"embedding model name {body.embedding_model.full_name}",
        )
    return info


@router.get(
    "/collections/get-blue-info",
    status_code=status.HTTP_200_OK,
)
def get_blue_collection_info():
    collection = context.vectordb.get_blue_collection()
    if not collection:
        msg = "Blue collection not found"
        logger.debug(msg)
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=msg)

    info = collection.get_state_info()
    logger.debug(f"Blue collection: {info}")
    return info


@router.post(
    "/collections/objects/insert",
    status_code=status.HTTP_200_OK,
)
def insert_objects(body: InsertObjectsRequest):
    collection = get_collection(body.embedding_model)
    collection.insert(body.objects)


@router.post(
    "/collections/objects/upsert",
    status_code=status.HTTP_200_OK,
)
def upsert_objects(body: UpsertObjectsRequest):
    collection = get_collection(body.embedding_model)
    collection.upsert(objects=body.objects, shrink_parts=body.shrink_parts)


@router.post(
    "/collections/objects/delete",
    status_code=status.HTTP_200_OK,
)
def delete_objects(body: DeleteObjectRequest):
    collection = get_collection(body.embedding_model)
    collection.delete(body.object_ids)


@router.post(
    "/collections/objects/find-by-ids",
    status_code=status.HTTP_200_OK,
)
def find_objects_by_ids(body: FindObjectsByIdsRequest):
    collection = get_collection(body.embedding_model)
    objects = collection.find_by_ids(body.object_ids)
    logger.debug(f"Found objects: {objects}")
    return objects


@router.post(
    "/collections/objects/find-similar",
    status_code=status.HTTP_200_OK,
)
def find_similar_objects(body: FindSimilarObjectsRequest):
    collection = get_collection(body.embedding_model)
    objects = collection.find_similarities(
        query_vector=body.query_vector,
        limit=body.limit,
        offset=body.offset,
        max_distance=body.max_distance,
    )
    logger.debug(f"Found similar objects: {objects}")
    return objects
