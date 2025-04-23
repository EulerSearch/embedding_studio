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
from embedding_studio.utils.plugin_utils import get_vectordb
from embedding_studio.vectordb.exceptions import CollectionNotFoundError
from embedding_studio.vectordb.vectordb import VectorDb

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/collections/create",
    status_code=status.HTTP_200_OK,
)
def create_collection(body: CreateCollectionRequest):
    """
    Creates a new vector collection for the specified embedding model.

    Validates the embedding model exists, then creates both a primary collection
    and a query-optimized collection for efficient searching. Returns collection
    state information upon successful creation.

    If the model doesn't exist, returns a 404 error with appropriate details.
    """
    iteration = context.mlflow_client.get_iteration_by_id(
        body.embedding_model_id
    )
    if iteration is None:
        message = (
            f"Fine tuning iteration with ID"
            + f"[{body.embedding_model_id}] does not exist."
        )
        logger.error(
            f"Something went wrong during collection deletion: {message}"
        )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=message
        )

    plugin = context.plugin_manager.get_plugin(iteration.plugin_name)
    vectordb = get_vectordb(plugin)

    collection = vectordb.create_collection(body.embedding_model)
    info = collection.get_state_info()
    logger.debug(f"Collection created: {info.model_dump()}")
    query_collection = vectordb.create_query_collection(body.embedding_model)
    info = query_collection.get_state_info()
    logger.debug(f"Query collection created: {info.model_dump()}")

    return info


@router.post(
    "/collections/create-index",
    status_code=status.HTTP_200_OK,
)
def create_index(body: CreateIndexRequest):
    """
    Builds search indexes on an existing vector collection.

    Creates HNSW graph indexes for both the main collection and its query-optimized
    variant. This step is essential for enabling efficient similarity searches and
    should be called after collection creation but before performing searches.
    """
    collection = context.vectordb.get_collection(body.embedding_model_id)
    collection.create_index()

    query_collection = context.vectordb.get_query_collection(
        body.embedding_model_id
    )
    query_collection.create_index()


@router.post(
    "/collections/categories/create-index",
    status_code=status.HTTP_200_OK,
)
def create_categories_index(body: CreateIndexRequest):
    """
    Builds search indexes on a categories-specific vector collection.

    Works with the specialized categories vector database to create indexes for
    both main and query collections. Follows the same pattern as regular index
    creation but optimized for category-based vector operations.
    """
    collection = context.categories_vectordb.get_collection(
        body.embedding_model_id
    )
    collection.create_index()

    query_collection = context.categories_vectordb.get_query_collection(
        body.embedding_model_id
    )
    query_collection.create_index()


def _delete_collection(vectordb: VectorDb, collection_id: str):
    try:
        collection = vectordb.get_collection(collection_id)
        info = collection.get_state_info()
        logger.debug(f"Delete collection: {info}")
        vectordb.delete_collection(collection_id)

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
        query_collection = vectordb.get_query_collection(collection_id)
        info = query_collection.get_state_info()
        logger.debug(f"Delete query collection: {info}")
        vectordb.delete_query_collection(collection_id)

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


@router.post(
    "/collections/delete",
    status_code=status.HTTP_200_OK,
)
def delete_collection(body: DeleteCollectionRequest):
    """
    Removes a vector collection and its associated resources.

    Deletes both the main collection and its query-optimized variant to provide
    proper resource cleanup. Returns 404 if the collection doesn't exist or
    detailed error information if deletion fails.
    """
    return _delete_collection(context.vectordb, body.embedding_model_id)


@router.post(
    "/collections/categories/delete",
    status_code=status.HTTP_200_OK,
)
def delete_categories_collection(body: DeleteCollectionRequest):
    """
    Removes a categories-specific vector collection.

    Works with the specialized categories vector database to remove category
    vector collections and their query variants. Ensures proper cleanup of
    category vector data and returns appropriate error details on failure.
    """
    return _delete_collection(
        context.categories_vectordb, body.embedding_model_id
    )


@router.get(
    "/collections/list",
    status_code=status.HTTP_200_OK,
)
def list_collections():
    """
    Lists all available vector collections in the database.

    Returns comprehensive metadata for all collections including their state,
    index status, and embedding model details. Provides an overview of available
    vector storage resources. Returns an empty list if no collections exist.
    """
    collection_infos = context.vectordb.list_collections()
    logger.debug(f"Found collections: {collection_infos}")
    return ListCollectionsResponse(collections=collection_infos)


@router.get(
    "/collections/queries/list",
    status_code=status.HTTP_200_OK,
)
def list_query_collections():
    """
    Lists all query-optimized collections in the database.

    Returns metadata for all query collections to help monitor the query
    optimization infrastructure.
    """
    collection_infos = context.vectordb.list_query_collections()
    logger.debug(f"Found query collections: {collection_infos}")
    return ListCollectionsResponse(collections=collection_infos)


@router.get(
    "/collections/categories/list",
    status_code=status.HTTP_200_OK,
)
def list_category_collections():
    """
    Lists all category-specific collections in the database.

    Works with the specialized categories vector database to return metadata
    for all category collections.
    """
    collection_infos = context.categories_vectordb.list_collections()
    logger.debug(f"Found category collections: {collection_infos}")
    return ListCollectionsResponse(collections=collection_infos)


@router.get(
    "/collections/get-info",
    status_code=status.HTTP_200_OK,
)
def get_collection_info(body: GetCollectionInfoRequest):
    """
    Retrieves detailed information about a specific collection.

    Returns comprehensive collection state and configuration for the specified
    embedding model ID. Returns a 404 error if the collection doesn't exist.
    """
    try:
        collection = context.vectordb.get_collection(body.embedding_mode_id)
        info = collection.get_state_info()
        logger.debug(f"Found collection: {info}")
        return info
    except CollectionNotFoundError:
        msg = f"Collection with id={body.embedding_model_id} not found"
        logger.debug(msg)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=msg,
        )


@router.get(
    "/collections/categories/get-info",
    status_code=status.HTTP_200_OK,
)
def get_categories_collection_info(body: GetCollectionInfoRequest):
    """
    Retrieves detailed information about a category-specific collection.

    Works with the specialized categories vector database to return detailed
    collection state information. Returns a 404 error if the collection doesn't
    exist. Follows the same pattern as regular collection info retrieval.
    """
    try:
        collection = context.categories_vectordb.get_collection(
            body.embedding_model.id
        )
        info = collection.get_state_info()
        logger.debug(f"Found collection: {info}")
        return info
    except CollectionNotFoundError:
        msg = f"Collection with id={body.embedding_model.id} not found"
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
    """
    Promotes a collection to "blue" (active/primary) status.

    Implements blue-green deployment pattern for zero-downtime updates by
    designating which collection serves as the primary for operations. Returns the
    collection state after the change or 404 if the collection doesn't exist.
    """
    try:
        collection = context.vectordb.get_collection(body.embedding_model_id)
        info = collection.get_state_info()

        logger.debug(f"Blue collection set: {info}")
        context.vectordb.set_blue_collection(
            embedding_model_id=info.embedding_model.id
        )
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


@router.post(
    "/collections/categories/set-blue",
    status_code=status.HTTP_200_OK,
)
def set_blue_categories_collection(body: SetBlueCollectionRequest):
    """
    Promotes a category collection to "blue" (active/primary) status.

    Works with the specialized categories vector database to implement blue-green
    deployment for category vectors. Enables zero-downtime updates and returns
    appropriate error details on failure.
    """
    try:
        collection = context.categories_vectordb.get_collection(
            body.embedding_model_id
        )
        info = collection.get_state_info()

        logger.debug(f"Blue collection set: {info}")
        context.categories_vectordb.set_blue_collection(
            embedding_model_id=info.embedding_model.id
        )
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
    """
    Retrieves information about the current blue (active) collection.

    Returns detailed state information about the primary collection currently
    serving production traffic. Returns a 404 error if no blue collection has
    been designated.
    """
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
    """
    Retrieves information about the blue (active) query collection.

    Returns state information about the primary query collection to help monitor
    the query optimization infrastructure. Returns a 404 error if no blue query
    collection exists.
    """
    collection = context.vectordb.get_blue_query_collection()
    if not collection:
        msg = "Blue collection not found"
        logger.debug(msg)
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=msg)

    info = collection.get_state_info()
    logger.debug(f"Blue collection: {info}")
    return info


@router.get(
    "/collections/categories/get-blue-info",
    status_code=status.HTTP_200_OK,
)
def get_blue_category_collection_info():
    """
    Retrieves information about the blue (active) category collection.

    Works with the specialized categories vector database to return detailed state
    about the primary category collection. Returns a 404 error if no blue category
    collection exists.
    """
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
    """
    Adds new vector objects to a collection.

    Optimized for adding new objects that don't exist yet. Does not update
    existing objects with the same IDs. Used for initial data loading scenarios
    when objects are being created for the first time.
    """
    collection = context.vectordb.get_collection(
        embedding_model_id=body.embedding_model_id
    )
    collection.insert(body.objects)


@router.post(
    "/collections/categories/objects/insert",
    status_code=status.HTTP_200_OK,
)
def insert_categories_objects(body: InsertObjectsRequest):
    """
    Adds new category vector objects to a collection.

    Works with the specialized categories vector database to insert category
    objects. Follows the same pattern as regular object insertion but optimized
    for category-specific vector data.
    """
    collection = context.categories_vectordb.get_collection(
        embedding_model_id=body.embedding_model_id
    )
    collection.insert(body.objects)


@router.post(
    "/collections/objects/upsert",
    status_code=status.HTTP_200_OK,
)
def upsert_objects(body: UpsertObjectsRequest):
    """
    Adds or updates vector objects in a collection.

    Supports both insertion of new objects and updates to existing ones. Includes
    option to optimize vector parts during the operation. The central operation
    for maintaining vector data over time.
    """
    collection = context.vectordb.get_collection(
        embedding_model_id=body.embedding_model_id
    )
    collection.upsert(objects=body.objects, shrink_parts=body.shrink_parts)


@router.post(
    "/collections/categories/objects/upsert",
    status_code=status.HTTP_200_OK,
)
def upsert_categories_objects(body: UpsertObjectsRequest):
    """
     Adds or updates category vector objects in a collection.

    Works with the specialized categories vector database to upsert category
    objects.
    """
    collection = context.categories_vectordb.get_collection(
        embedding_model_id=body.embedding_model_id
    )
    collection.upsert(objects=body.objects, shrink_parts=body.shrink_parts)


@router.post(
    "/collections/objects/delete",
    status_code=status.HTTP_200_OK,
)
def delete_objects(body: DeleteObjectRequest):
    """
    Removes specific vector objects from a collection.

    Supports targeted cleanup of obsolete vector data by efficiently removing
    objects by ID without scanning.
    """
    collection = context.vectordb.get_collection(body.embedding_model_id)
    collection.delete(body.object_ids)


@router.post(
    "/collections/categories/objects/delete",
    status_code=status.HTTP_200_OK,
)
def delete_categories_objects(body: DeleteObjectRequest):
    """
    Removes specific category vector objects from a collection.

    Works with the specialized categories vector database to efficiently remove
    category vector data by ID.
    """
    collection = context.categories_vectordb.get_collection(
        body.embedding_model_id
    )
    collection.delete(body.object_ids)


@router.post(
    "/collections/objects/find-by-ids",
    status_code=status.HTTP_200_OK,
)
def find_objects_by_ids(body: FindObjectsByIdsRequest):
    """
    Retrieves vector objects by their identifiers.

    Performs exact-match retrieval by ID without vector search. Returns complete
    object data including vectors and metadata.
    """
    collection = context.vectordb.get_collection(body.embedding_model_id)
    objects = collection.find_by_ids(body.object_ids)
    logger.debug(f"Found objects: {objects}")
    return objects


@router.post(
    "/collections/categories/objects/find-by-ids",
    status_code=status.HTTP_200_OK,
)
def find_categories_objects_by_ids(body: FindObjectsByIdsRequest):
    """
    Retrieves category vector objects by their identifiers.

    Works with the specialized categories vector database to perform exact
    ID-based lookups. Returns complete category object data including vectors
    and metadata for category management operations.
    """
    collection = context.categories_vectordb.get_collection(
        body.embedding_model_id
    )
    objects = collection.find_by_ids(body.object_ids)
    logger.debug(f"Found objects: {objects}")
    return objects


@router.post(
    "/collections/objects/find-similar",
    status_code=status.HTTP_200_OK,
)
def find_similar_objects(body: FindSimilarObjectsRequest):
    """
    Performs vector similarity search to find related objects.

    Requires a query vector to compare against stored vectors, with support for
    limit, offset, and maximum distance parameters. Returns objects sorted by
    similarity (closest first) for semantic search capabilities.
    """
    collection = context.vectordb.get_collection(body.embedding_model_id)
    objects = collection.find_similarities(
        query_vector=body.query_vector,
        limit=body.limit,
        offset=body.offset,
        max_distance=body.max_distance,
    )
    logger.debug(f"Found similar objects: {objects}")
    return objects


@router.post(
    "/collections/categories/objects/find-similar",
    status_code=status.HTTP_200_OK,
)
def find_similar_categories_objects(body: FindSimilarObjectsRequest):
    """
    Performs similarity search on category vectors.

    Works with the specialized categories vector database to find similar
    categories using the same parameters as regular similarity search.
    """
    collection = context.categories_vectordb.get_collection(
        body.embedding_model_id
    )
    objects = collection.find_similarities(
        query_vector=body.query_vector,
        limit=body.limit,
        offset=body.offset,
        max_distance=body.max_distance,
    )
    logger.debug(f"Found similar objects: {objects}")
    return objects
