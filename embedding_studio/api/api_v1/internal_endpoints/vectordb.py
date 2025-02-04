import logging

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
from embedding_studio.utils.plugin_utils import (
    get_vectordb,
    get_vectordb_by_fine_tuning_name,
)
from embedding_studio.vectordb.exceptions import CollectionNotFoundError

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/collections/create",
    status_code=status.HTTP_200_OK,
)
def create_collection(body: CreateCollectionRequest):
    plugin = context.plugin_manager.get_plugin(body.embedding_model.name)
    search_index_info = plugin.get_search_index_info()
    vectordb = get_vectordb(plugin)

    collection = vectordb.create_collection(
        body.embedding_model, search_index_info
    )
    info = collection.get_state_info()
    logger.debug(f"Collection created: {info.model_dump()}")
    query_collection = vectordb.create_query_collection(
        body.embedding_model, search_index_info
    )
    info = query_collection.get_state_info()
    logger.debug(f"Query collection created: {info.model_dump()}")

    return info


@router.post(
    "/collections/create-index",
    status_code=status.HTTP_200_OK,
)
def create_index(body: CreateIndexRequest):
    vectordb = get_vectordb_by_fine_tuning_name(body.embedding_model.name)

    collection = vectordb.get_collection(body.embedding_model)
    collection.create_index()

    query_collection = vectordb.get_query_collection(body.embedding_model)
    query_collection.create_index()


@router.post(
    "/collections/delete",
    status_code=status.HTTP_200_OK,
)
def delete_collection(body: DeleteCollectionRequest):
    vectordb = get_vectordb_by_fine_tuning_name(body.embedding_model.name)

    try:
        collection = vectordb.get_collection(body.embedding_model)
        info = collection.get_state_info()
        logger.debug(f"Delete collection: {info}")
        vectordb.delete_collection(body.embedding_model)

    except CollectionNotFoundError as err:
        logger.debug(f"Collection not found: {err}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"{err}"
        )

    except Exception as err:
        logger.exception(
            f"Something went wrong during collection deletion: {err}"
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"{err}"
        )

    # Deleting related query collection
    try:
        query_collection = vectordb.get_query_collection(body.embedding_model)
        info = query_collection.get_state_info()
        logger.debug(f"Delete query collection: {info}")
        vectordb.delete_query_collection(body.embedding_model)

    except CollectionNotFoundError as err:
        logger.debug(f"Query collection not found: {err}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"{err}"
        )

    except Exception as err:
        logger.exception(
            f"Something went wrong during query collection deletion: {err}"
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"{err}",
        )


@router.get(
    "/collections/list",
    status_code=status.HTTP_200_OK,
)
def list_collections():
    collection_infos = context.vectordb.list_collections()
    logger.debug(f"Found collections: {collection_infos}")
    return ListCollectionsResponse(collections=collection_infos)


@router.get(
    "/collections/queries/list",
    status_code=status.HTTP_200_OK,
)
def list_query_collections():
    collection_infos = context.vectordb.list_query_collections()
    logger.debug(f"Found query collections: {collection_infos}")
    return ListCollectionsResponse(collections=collection_infos)


@router.get(
    "/collections/categories/list",
    status_code=status.HTTP_200_OK,
)
def list_category_collections():
    collection_infos = context.categories_vectordb.list_collections()
    logger.debug(f"Found category collections: {collection_infos}")
    return ListCollectionsResponse(collections=collection_infos)


@router.get(
    "/collections/get-info",
    status_code=status.HTTP_200_OK,
)
def get_collection_info(body: GetCollectionInfoRequest):
    try:
        vectordb = get_vectordb_by_fine_tuning_name(body.embedding_model.name)

        collection = vectordb.get_collection(body.embedding_model)
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
        vectordb = get_vectordb_by_fine_tuning_name(body.embedding_model.name)

        collection = vectordb.get_collection(body.embedding_model)
        info = collection.get_state_info()

        logger.debug(f"Blue collection set: {info}")
        vectordb.set_blue_collection(embedding_model=info.embedding_model)
    except CollectionNotFoundError as err:
        logger.debug(f"Collection is not found: {err}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Collection is not found",
        )

    except Exception as err:
        logger.exception(
            f"Something went wrong during collection set blue: {err}"
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Something went wrong during collection set blue",
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


@router.get(
    "/collections/get-blue-query-info",
    status_code=status.HTTP_200_OK,
)
def get_blue_query_collection_info():
    collection = context.vectordb.get_blue_query_collection()
    if not collection:
        msg = "Blue collection not found"
        logger.debug(msg)
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=msg)

    info = collection.get_state_info()
    logger.debug(f"Blue collection: {info}")
    return info


@router.get(
    "/collections/get-category-info",
    status_code=status.HTTP_200_OK,
)
def get_blue_category_collection_info():
    collection = context.categories_vectordb.get_blue_collection()
    if not collection:
        msg = "Blue category collection not found"
        logger.debug(msg)
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=msg)

    info = collection.get_state_info()
    logger.debug(f"Blue category collection: {info}")
    return info


@router.post(
    "/collections/objects/insert",
    status_code=status.HTTP_200_OK,
)
def insert_objects(body: InsertObjectsRequest):
    vectordb = get_vectordb_by_fine_tuning_name(body.embedding_model.name)

    collection = vectordb.get_collection(body.embedding_model)
    collection.insert(body.objects)


@router.post(
    "/collections/objects/upsert",
    status_code=status.HTTP_200_OK,
)
def upsert_objects(body: UpsertObjectsRequest):
    vectordb = get_vectordb_by_fine_tuning_name(body.embedding_model.name)

    collection = vectordb.get_collection(body.embedding_model)
    collection.upsert(objects=body.objects, shrink_parts=body.shrink_parts)


@router.post(
    "/collections/objects/delete",
    status_code=status.HTTP_200_OK,
)
def delete_objects(body: DeleteObjectRequest):
    vectordb = get_vectordb_by_fine_tuning_name(body.embedding_model.name)

    collection = vectordb.get_collection(body.embedding_model)
    collection.delete(body.object_ids)


@router.post(
    "/collections/objects/find-by-ids",
    status_code=status.HTTP_200_OK,
)
def find_objects_by_ids(body: FindObjectsByIdsRequest):
    vectordb = get_vectordb_by_fine_tuning_name(body.embedding_model.name)

    collection = vectordb.get_collection(body.embedding_model)
    objects = collection.find_by_ids(body.object_ids)
    logger.debug(f"Found objects: {objects}")
    return objects


@router.post(
    "/collections/objects/find-similar",
    status_code=status.HTTP_200_OK,
)
def find_similar_objects(body: FindSimilarObjectsRequest):
    vectordb = get_vectordb_by_fine_tuning_name(body.embedding_model.name)

    collection = vectordb.get_collection(body.embedding_model)
    objects = collection.find_similarities(
        query_vector=body.query_vector,
        limit=body.limit,
        offset=body.offset,
        max_distance=body.max_distance,
    )
    logger.debug(f"Found similar objects: {objects}")
    return objects
