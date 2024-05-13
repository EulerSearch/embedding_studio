import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, status

from embedding_studio.api.api_v1.schemas.vectrordb import (
    CreateCollectionRequest,
    DeleteObjectRequest,
    FindObjectsByIdsRequest,
    FindSimilarObjectsRequest,
    InsertObjectsRequest,
    ListCollectionsResponse,
    UpsertObjectsRequest,
)
from embedding_studio.context.app_context import context
from embedding_studio.vectordb.exceptions import (
    CollectionNotFoundError,
    DeleteBlueCollectionError,
)

logger = logging.getLogger(__name__)

router = APIRouter()


def get_blue_collection():
    collection = context.vectordb.get_blue_collection()
    if not collection:
        msg = "Blue collection not found"
        logger.debug(msg)
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=msg)
    return collection


def get_collection(collection_id: Optional[str] = None):
    if not collection_id:
        return get_blue_collection()
    try:
        return context.vectordb.get_collection(collection_id)
    except CollectionNotFoundError:
        msg = f"Collection with id={collection_id} not found"
        logger.debug(msg)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=msg,
        )


@router.post(
    "/collections/create",
    status_code=status.HTTP_200_OK,
)
def create_collection(body: CreateCollectionRequest):
    collection = context.vectordb.create_collection(
        body.model, body.collection_id
    )
    info = collection.get_state_info()
    logger.debug(f"collection created: {info.model_dump()}")
    return info


@router.post(
    "/collections/create-index",
    status_code=status.HTTP_200_OK,
)
def create_index(collection_id: str):
    collection = get_collection(collection_id)
    collection.create_index()


@router.post(
    "/collections/delete",
    status_code=status.HTTP_200_OK,
)
def delete_collection(collection_id: str):
    try:
        collection = get_collection(collection_id)
        info = collection.get_state_info()
        logger.debug(f"Delete collection: {info}")
    except Exception:
        pass

    try:
        context.vectordb.delete_collection(collection_id)
    except CollectionNotFoundError as err:
        logger.warning(f"Can't find collection wile deleting: {err}")
        return
    except DeleteBlueCollectionError as err:
        raise HTTPException(
            status_code=status.HTTP_406_NOT_ACCEPTABLE, detail=f"{err}"
        )

    try:
        collection = context.vectordb.get_collection(collection_id)
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
def get_collection_info(collection_id: str):
    collection = get_collection(collection_id)
    info = collection.get_state_info()
    logger.debug(f"Found collection: {info}")
    return info


@router.post(
    "/collections/set-blue",
    status_code=status.HTTP_200_OK,
)
def set_blue_collection(collection_id: str):
    try:
        context.vectordb.set_blue_collection(collection_id)
    except CollectionNotFoundError as err:
        logger.debug(f"Collection not found: {err}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Collection not found",
        )
    collection = get_blue_collection()
    info = collection.get_state_info()
    logger.debug(f"Blue collection set: {info}")
    assert info.collection_id == collection_id
    return info


@router.get(
    "/collections/get-blue-info",
    status_code=status.HTTP_200_OK,
)
def get_blue_collection_info():
    collection = get_blue_collection()
    info = collection.get_state_info()
    logger.debug(f"Blue collection: {info}")
    return info


@router.post(
    "/collections/objects/insert",
    status_code=status.HTTP_200_OK,
)
def insert_objects(
    body: InsertObjectsRequest, collection_id: Optional[str] = None
):
    collection = get_collection(collection_id)
    collection.insert(body.objects)


@router.post(
    "/collections/objects/upsert",
    status_code=status.HTTP_200_OK,
)
def upsert_objects(
    body: UpsertObjectsRequest, collection_id: Optional[str] = None
):
    collection = get_collection(collection_id)
    collection.upsert(objects=body.objects, shrink_parts=body.shrink_parts)


@router.post(
    "/collections/objects/delete",
    status_code=status.HTTP_200_OK,
)
def delete_objects(
    body: DeleteObjectRequest, collection_id: Optional[str] = None
):
    collection = get_collection(collection_id)
    collection.delete(body.object_ids)


@router.post(
    "/collections/objects/find-by-ids",
    status_code=status.HTTP_200_OK,
)
def find_objects_by_ids(
    body: FindObjectsByIdsRequest, collection_id: Optional[str] = None
):
    collection = get_collection(collection_id)
    objects = collection.find_by_ids(body.object_ids)
    logger.debug(f"Found objects: {objects}")
    return objects


@router.post(
    "/collections/objects/find-similar",
    status_code=status.HTTP_200_OK,
)
def find_similar_objects(
    body: FindSimilarObjectsRequest, collection_id: Optional[str] = None
):
    collection = get_collection(collection_id)
    objects = collection.find_similarities(
        query_vector=body.query_vector,
        limit=body.limit,
        offset=body.offset,
        max_distance=body.max_distance,
    )
    logger.debug(f"Found similar objects: {objects}")
    return objects
