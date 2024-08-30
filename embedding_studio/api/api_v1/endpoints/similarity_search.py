import logging
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, status

from embedding_studio.api.api_v1.schemas.similarity_search import (
    SearchResult,
    SimilaritySearchRequest,
    SimilaritySearchResponse,
)
from embedding_studio.context.app_context import context
from embedding_studio.models.clickstream.sessions import (
    SearchResultItem,
    Session,
)
from embedding_studio.models.embeddings.objects import SearchResults
from embedding_studio.utils.datetime_utils import utc_timestamp

# Initialize logger for this module
logger = logging.getLogger(__name__)

# Initialize FastAPI router for handling API endpoints
router = APIRouter()


def _find_similars(body: SimilaritySearchRequest) -> SearchResults:
    """
    Perform similarity search on the embeddings collection based on the query.

    :param body: Request body containing search parameters.
    :return: Search results containing similar objects.
    :raises HTTPException: If the search fails or the collection is not initialized.
    """
    # Retrieve the collection where embeddings are stored
    collection = context.vectordb.get_blue_collection()

    if not collection:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model is not initialized yet.",
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

    try:
        logger.debug("Retrieving search query.")
        # Retrieve and vectorize the search query
        search_query = query_retriever(body.search_query)

        logger.debug("Search query vectorizing.")
        query_vector = inference_client.forward_query(search_query)[0]

        logger.debug("Searching for similar objects.")
        # Search for similar objects in the collection
        search_results = collection.find_similarities(
            query_vector=query_vector,
            offset=body.offset,
            limit=body.limit,
            max_distance=body.max_distance,
            payload_filter=body.filter,
        )

        logger.debug(
            f"Found {len(search_results.found_objects)} similar objects."
        )

    except Exception:
        # Log and raise an HTTP exception if something goes wrong during the search
        logger.exception(
            "Something went wrong while searching for similar objects."
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Something went wrong while searching for similar objects.",
        )

    return search_results


def _create_session_object(
        body: SimilaritySearchRequest,
        session_id: str
) -> Session:
    """
    Create a new session placeholder.

    :param body: Request body containing session and search information.
    :param session_id: Specified session id.
    :return: Existing session if found, otherwise a placeholder for a new session.
    """


    return Session(
        session_id=session_id,
        search_query=body.search_query,
        user_id=body.user_id,
        search_results=[],
        created_at=utc_timestamp(),
    )


def _register_session_with_results(
    session: Session, search_results: SearchResults
) -> None:
    """
    Update the session with search results and register the session.

    :param session: The session object to be updated and registered.
    :param search_results: The results of the similarity search to be added to the session.
    :raises HTTPException: If session registration fails.
    """
    # Update the session with the new search results
    session.search_results = [
        SearchResultItem(
            object_id=found_object.object_id,
            rank=found_object.distance,
            meta=found_object.storage_meta,
        )
        for found_object in search_results.found_objects
    ]

    try:
        # Register or update the session, ensuring idempotency and consistency
        reg_session = context.clickstream_dao.register_session(session)
        session.session_id = reg_session.session_id
    except Exception:
        logger.exception("Something went wrong while registering the session.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Something went wrong while registering the session.",
        )


@router.post(
    "/similarity-search",
    response_model=SimilaritySearchResponse,
    response_model_by_alias=False,
    response_model_exclude_none=True,
)
def similarity_search(body: SimilaritySearchRequest) -> Any:
    """
    Endpoint to search for similar objects based on the provided query.

    :param body: Request body containing the similarity search parameters.
    :return: Response containing the session ID and search results.
    """
    logger.debug(f"POST /embeddings/similarity-search: {body}")
    # Perform similarity search as the session was not found or has no results
    search_results = _find_similars(body)

    session_id = None
    if body.create_session:
        # Generate a new session if not found
        session_id = body.session_id if body.session_id is not None else str(uuid.uuid4())
        # Check for an existing session first
        session = _create_session_object(body, session_id)
        # Register the session with the search results
        _register_session_with_results(session, search_results)

    # Return the search results along with the session ID
    return SimilaritySearchResponse(
        session_id=session_id,
        search_results=[
            SearchResult(
                object_id=found_object.object_id,
                distance=found_object.distance,
                payload=found_object.payload,
                meta=found_object.storage_meta,
            )
            for found_object in search_results.found_objects
        ],
        next_page_offset=search_results.next_offset,
    )
