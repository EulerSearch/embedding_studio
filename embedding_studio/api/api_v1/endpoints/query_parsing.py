import logging
from typing import Any, List

from fastapi import APIRouter, HTTPException, status

from embedding_studio.api.api_v1.schemas.query_parsing import (
    QueryParsingCategoriesResponse,
    QueryParsingRequest,
)
from embedding_studio.api.api_v1.schemas.similarity_search import SearchResult
from embedding_studio.context.app_context import context
from embedding_studio.core.config import settings
from embedding_studio.models.embeddings.objects import SearchResults

# Initialize logger for this module
logger = logging.getLogger(__name__)

# Initialize FastAPI router for handling API endpoints
router = APIRouter()


def _get_similar_categories(search_query: Any) -> List[SearchResults]:
    # Retrieve the collection where embeddings are stored
    collection = context.categories_vectordb.get_blue_collection()

    if not collection:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Category model is not initialized yet.",
        )

    # Get collection info and the associated embedding model plugin
    collection_info = collection.get_info()
    plugin = context.plugin_manager.get_plugin(
        collection_info.embedding_model.name
    )

    # Retrieve query retriever and inference client from the plugin
    query_retriever = plugin.get_query_retriever()

    inference_client = plugin.get_inference_client_factory().get_client(
        collection_info.embedding_model.id
    )

    categories_selector = plugin.get_category_selector()

    try:
        logger.debug("Retrieving search query.")
        # Retrieve and vectorize the search query
        search_query = query_retriever(search_query)
        logger.debug("Search query vectorizing.")
        query_vector = inference_client.forward_query(search_query)[0]
        logger.debug("Searching for similar categories.")

        # Search for similar objects in the collection
        found_objects, _ = collection.find_similar_objects(
            query_vector=query_vector.tolist(),
            offset=0,
            limit=plugin.get_max_similar_categories(),
            max_distance=plugin.get_max_margin(),
            with_vectors=categories_selector.vectors_are_needed,
            meta_info=settings.QUERY_PARSING_DB_META_INFO,
        )
        logger.debug(f"Found {len(found_objects)} similar categories.")

    except Exception:
        # Log and raise an HTTP exception if something goes wrong during the search
        logger.exception(
            "Something went wrong while searching for similar objects."
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Something went wrong while searching for similar category objects.",
        )

    if len(found_objects) == 0:
        return []

    final_indexes = categories_selector.select(found_objects, query_vector)
    results = []
    for index in final_indexes:
        results.append(found_objects[index])

    print(results)
    return results


@router.post(
    "/categories",
    response_model=QueryParsingCategoriesResponse,
    response_model_by_alias=False,
    response_model_exclude_none=True,
)
def parse_categories(body: QueryParsingRequest) -> Any:
    """
    Endpoint to search for relevant categories based on the provided query.

    :param body: Request body containing the search query being parsed.
    :return: Response containing similar categories.
    """
    logger.debug(f"POST /parse-query/categories: {body}")
    similar_categories = _get_similar_categories(body.search_query)

    return QueryParsingCategoriesResponse(
        categories=[
            SearchResult(
                object_id=found_object.object_id,
                distance=found_object.distance,
                payload=found_object.payload,
                meta=found_object.storage_meta,
            )
            for found_object in similar_categories
        ],
    )
