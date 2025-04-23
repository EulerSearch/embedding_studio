import logging
import uuid
from typing import Any, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException, status

from embedding_studio.api.api_v1.schemas.similarity_search import (
    CountResponse,
    PayloadCountRequest,
    PayloadSearchRequest,
    SearchResult,
    SimilaritySearchRequest,
    SimilaritySearchResponse,
)
from embedding_studio.context.app_context import context
from embedding_studio.models.clickstream.sessions import (
    SearchResultItem,
    Session,
)
from embedding_studio.models.embeddings.objects import (
    Object,
    ObjectPart,
    SearchResults,
    SimilarObject,
)
from embedding_studio.models.payload.models import PayloadFilter
from embedding_studio.models.sort_by.models import SortByOptions
from embedding_studio.utils.datetime_utils import utc_timestamp

# Initialize logger for this module
logger = logging.getLogger(__name__)

# Initialize FastAPI router for handling API endpoints
router = APIRouter()


def _find_by_payload_fiter(body: SimilaritySearchRequest) -> SearchResults:
    # Retrieve the collection where embeddings are stored
    collection = context.vectordb.get_blue_collection()

    if not collection:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model is not initialized yet.",
        )

    logger.debug("Searching for objects filtered by payload.")
    # Search for similar objects in the collection
    search_results = collection.find_by_payload_filter(
        offset=body.offset,
        limit=body.limit,
        payload_filter=PayloadFilter.model_validate(body.filter.model_dump())
        if body.filter
        else None,
        sort_by=body.sort_by,
    )

    logger.debug(
        f"Found {len(search_results.found_objects)} filtered by payload objects."
    )
    return search_results


def _count_by_payload_filter(body: PayloadCountRequest) -> int:
    # Retrieve the collection where embeddings are stored
    collection = context.vectordb.get_blue_collection()

    if not collection:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model is not initialized yet.",
        )

    total_count = collection.count_by_payload_filter(
        payload_filter=PayloadFilter.model_validate(body.filter.model_dump())
        if body.filter
        else None,
    )

    return total_count


def _find_similars(
    body: SimilaritySearchRequest, background_tasks: BackgroundTasks
) -> SearchResults:
    """
    Perform similarity search on the embeddings collection based on the query.

    :param body: Request body containing search parameters.
    :param background_tasks: FastAPI BackgroundTasks instance to schedule tasks.
    :return: Search results containing similar objects.
    :raises HTTPException: If the search fails or the collection is not initialized.
    """
    # Retrieve the collection where embeddings are stored
    collection = context.vectordb.get_blue_collection()
    query_collection = context.vectordb.get_blue_query_collection()

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
        query_vector = inference_client.forward_query(search_query)[0].tolist()

        async def insert_query_vector():
            """
            Insert the query vector into the query collection asynchronously.
            """
            try:
                logger.debug("Inserting query vector asynchronously.")
                object_id = f"{body.session_id}:{query_retriever.get_id(body.search_query)}"
                storage_meta = query_retriever.get_storage_metadata(
                    body.search_query
                )
                payload = query_retriever.get_payload(body.search_query)
                if payload is None:
                    payload = dict()

                query_collection.insert(
                    [
                        Object(
                            object_id=object_id,
                            payload=payload,
                            storage_meta=storage_meta,
                            parts=[
                                ObjectPart(
                                    part_id=f"{object_id}:0",
                                    vector=query_vector,
                                )
                            ],
                            user_id=body.user_id,
                            session_id=body.session_id,
                        )
                    ]
                )
                logger.debug("Query vector insertion completed.")
            except Exception as e:
                logger.error(f"Failed to insert query vector: {e}")

        if body.user_id:
            # Schedule the background task for query vector insertion
            background_tasks.add_task(insert_query_vector)

        logger.debug("Searching for similar objects.")
        # Search for similar objects in the collection
        search_results = collection.find_similarities(
            query_vector=query_vector,
            offset=body.offset,
            limit=body.limit,
            max_distance=body.max_distance,
            payload_filter=PayloadFilter.model_validate(
                body.filter.model_dump()
            )
            if body.filter
            else None,
            sort_by=body.sort_by,
            user_id=body.user_id,
            similarity_first=body.similarity_first,
            meta_info=body.meta_info,
        )

        logger.debug(
            f"Found {len(search_results.found_objects)} similar objects."
        )

        return search_results

    except Exception:
        # Log and raise an HTTP exception if something goes wrong during the search
        logger.exception(
            "Something went wrong while searching for similar objects."
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Something went wrong while searching for similar objects.",
        )


def _create_session_object(
    body: SimilaritySearchRequest,
    session_id: str,
    is_payload_search: bool = False,
    payload_filter: Optional[PayloadFilter] = None,
    sort_by: Optional[SortByOptions] = None,
) -> Session:
    """
    Create a new session placeholder.

    :param body: Request body containing session and search information.
    :param session_id: Specified session id.
    :param is_payload_search: Specified if search done via payload.
    :param payload_filter: Specified filter for payload search.
    :param sort_by: Specified sort by.
    :return: Existing session if found, otherwise a placeholder for a new session.
    """

    return Session(
        session_id=session_id,
        search_query=body.search_query,
        user_id=body.user_id,
        search_results=[],
        created_at=utc_timestamp(),
        is_payload_search=is_payload_search,
        payload_filter=PayloadFilter.model_validate(
            payload_filter.model_dump()
        )
        if body.filter
        else None,
        sort_by=SortByOptions.model_validate(sort_by.model_dump())
        if sort_by
        else None,
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
            rank=found_object.distance
            if isinstance(found_object, SimilarObject)
            else 1.0,
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
def similarity_search(
    body: SimilaritySearchRequest, background_tasks: BackgroundTasks
) -> Any:
    """
    Endpoint to search for similar objects based on the provided query.

    :param body: Request body containing the similarity search parameters.
    :param background_tasks: FastAPI BackgroundTasks instance to schedule tasks.
    :return: Response containing the session ID and search results.
    """
    logger.debug(f"POST /embeddings/similarity-search: {body}")

    session_id = None
    if body.create_session:
        # Generate a new session if not found
        session_id = (
            body.session_id
            if body.session_id is not None
            else str(uuid.uuid4())
        )

    body.session_id = session_id

    if body.search_query is None and body.filter is not None:
        search_results = _find_by_payload_fiter(body)

    elif body.search_query is None:
        search_results = SearchResults(found_objects=[], next_offset=0)

    else:
        # Perform similarity search as the session was not found or has no results
        search_results = _find_similars(body, background_tasks)

    if body.create_session:
        # Check for an existing session first
        session = _create_session_object(
            body,
            session_id,
            is_payload_search=False,
            payload_filter=body.filter,
            sort_by=body.sort_by,
        )
        # Register the session with the search results
        _register_session_with_results(session, search_results)

    # Return the search results along with the session ID
    return SimilaritySearchResponse(
        session_id=session_id,
        search_results=[
            SearchResult(
                object_id=found_object.object_id,
                distance=found_object.distance
                if isinstance(found_object, SimilarObject)
                else 1.0,
                payload=found_object.payload,
                meta=found_object.storage_meta,
            )
            for found_object in search_results.found_objects
        ],
        next_page_offset=search_results.next_offset,
        meta_info=search_results.meta_info,
    )


@router.post(
    "/payload-search",
    response_model=SimilaritySearchResponse,
    response_model_by_alias=False,
    response_model_exclude_none=True,
)
def payload_search(body: PayloadSearchRequest) -> Any:
    """
    Endpoint to search for similar objects based on the provided payload.

    :param body: Request body containing the payload search parameters.
    :return: Response containing the session ID and search results.
    """
    logger.debug(f"POST /embeddings/payload-search: {body}")

    search_results = _find_by_payload_fiter(body)

    session_id = None
    if body.create_session:
        # Generate a new session if not found
        session_id = (
            body.session_id
            if body.session_id is not None
            else str(uuid.uuid4())
        )
        # Check for an existing session first
        session = _create_session_object(
            body,
            session_id,
            is_payload_search=True,
            payload_filter=body.filter,
            sort_by=body.sort_by,
        )
        # Register the session with the search results
        _register_session_with_results(session, search_results)

    # Return the search results along with the session ID
    return SimilaritySearchResponse(
        session_id=session_id,
        search_results=[
            SearchResult(
                object_id=found_object.object_id,
                distance=found_object.distance
                if isinstance(found_object, SimilarObject)
                else 1.0,
                payload=found_object.payload,
                meta=found_object.storage_meta,
            )
            for found_object in search_results.found_objects
        ],
        next_page_offset=search_results.next_offset,
        total_count=search_results.total_count,
    )


@router.post(
    "/payload-count",
    response_model=CountResponse,
    response_model_by_alias=False,
    response_model_exclude_none=True,
)
def count_payload(body: PayloadCountRequest) -> Any:
    """
    Endpoint to count for similar objects based on the provided payload.

    :param body: Request body containing the payload search parameters.
    :return: Response containing total count.
    """
    logger.debug(f"POST /embeddings/payload-count: {body}")

    total_count = _count_by_payload_filter(body)

    return CountResponse(total_count=total_count)
