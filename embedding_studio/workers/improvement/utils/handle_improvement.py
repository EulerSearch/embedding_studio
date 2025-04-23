import logging
import traceback
from typing import List

import torch

from embedding_studio.context.app_context import context
from embedding_studio.core.config import settings
from embedding_studio.core.plugin import PluginManager
from embedding_studio.embeddings.data.clickstream.improvement_input import (
    ImprovementElement,
    ImprovementInput,
)
from embedding_studio.models.embeddings.objects import Object, ObjectPart
from embedding_studio.models.improvement import SessionForImprovementInDb
from embedding_studio.models.task import TaskStatus

logger = logging.getLogger(__name__)


def handle_improvement(
    sessions_for_improvement: List[SessionForImprovementInDb],
):
    """
    Process sessions for embedding vector improvement based on user interactions.

    This function takes a list of sessions marked for improvement and uses user interaction data
    (clicks) to adjust embedding vectors, making clicked results more relevant and non-clicked
    results less relevant for similar future queries.

    The process works as follows:
    1. Retrieve full session data for each improvement session
    2. Load the vector DB collections
    3. Group items by clicked/non-clicked status
    4. Normalize and prepare vectors for adjustment
    5. Call the vector adjustment algorithm
    6. Store improved vectors back to the database

    :param sessions_for_improvement: List of session objects marked for improvement
    :return: None - results are stored directly in the database
    """
    # List to store fully hydrated session objects with clickstream data
    sessions = []

    # Step 1: Retrieve full session data for each session ID
    for session in sessions_for_improvement:
        try:
            # Fetch complete session data including search results and click events
            sessions.append(
                context.clickstream_dao.get_session(session.session_id)
            )

        except Exception:
            # Log and mark as failed if session retrieval fails
            logger.exception(
                f"Something went wrong during retrieving session with ID: {session.session_id}"
            )
            session.status = TaskStatus.failed
            session.detail = traceback.format_exc()
            context.sessions_for_improvement.update(obj=session)

    # Initialize plugin manager to access vector adjustment implementation
    plugin_manager = PluginManager()
    # Load plugins from the configured directory
    plugin_manager.discover_plugins(directory=settings.ES_PLUGINS_PATH)

    try:
        # Step 2: Get the vector database collections
        # blue_collection: main document/content vectors
        # blue_query_collection: user query vectors
        blue_collection = context.vectordb.get_blue_collection()
        blue_query_collection = context.vectordb.get_blue_query_collection()

    except Exception:
        # Mark all sessions as failed if collection access fails
        logger.exception(
            f"Something went wrong during retrieving blue collection"
        )
        for session in sessions_for_improvement:
            session.status = TaskStatus.failed
            session.detail = traceback.format_exc()
            context.sessions_for_improvement.update(obj=session)

        return

    try:
        # Step 3: Prepare data structures for vector adjustment
        improvement_inputs = []  # Final list of inputs for the adjuster
        not_originals = (
            set()
        )  # Set of object IDs that are already personalized versions
        object_to_originals = dict()  # Maps personalized IDs to original IDs
        object_by_id = dict()  # Cache of objects by ID for quick lookup
        session_to_user = dict()  # Maps session IDs to user IDs

        # Process each session
        for session in sessions:
            # Store session to user mapping for later use
            session_to_user[session.session_id] = session.user_id

            # Get the query vector(s) for this session
            # A query vector represents what the user searched for
            query_object = blue_query_collection.get_objects_by_session_id(
                session.session_id
            )

            # Skip if no query found
            if len(query_object) == 0:
                continue

            # Take the first query object (typically only one per session)
            query_object = query_object[0]

            # Extract and combine query vector parts
            # Vector shape: [N, D] where N = number of parts, D = embedding dimension (typically 1024)
            query_vector = []
            for part in query_object.parts:
                query_vector.append(part.vector)
            query_vector = torch.Tensor(
                query_vector
            )  # Convert to torch tensor

            # Identify which items were clicked in this session
            clicked_object_ids = set()
            # Only process clicks if the session is marked as relevant
            if not session.is_irrelevant:
                for event in session.events:
                    clicked_object_ids.add(event.object_id)

            # Skip if no clicks recorded
            if len(clicked_object_ids) == 0:
                continue

            # Prepare containers for clicked and non-clicked items
            clicked_elements = []
            non_clicked_elements = []

            # Process each search result in the session
            for res in session.search_results:
                # Retrieve full vector information for this result
                res_obj = blue_collection.find_by_ids(
                    [
                        res.object_id,
                    ]
                )[0]

                # Track personalized vectors (non-original)
                if res_obj.original_id is not None:
                    not_originals.add(res_obj.object_id)
                    object_to_originals[
                        res_obj.object_id
                    ] = res_obj.original_id

                # Cache object for later use
                object_by_id[res_obj.object_id] = res_obj

                # Extract and combine result vector parts
                # Vector shape: [M, D] where M = number of parts, D = embedding dimension
                res_vector = []
                for part in res_obj.parts:
                    res_vector.append(part.vector)

                # Convert to torch tensor
                res_vector = torch.Tensor(res_vector)  # Shape: [M, D]

                # Create an improvement element with vector and metadata
                element = ImprovementElement(
                    id=res_obj.object_id,
                    vector=res_vector,  # Shape: [M, D]
                    # Track which parts are average vectors vs. specific part vectors
                    is_average=[
                        part.is_average if part.is_average else False
                        for part in res_obj.parts
                    ],
                    user_id=session.user_id,
                )

                # Sort into clicked or non-clicked based on user interaction
                if res.object_id in clicked_object_ids:
                    clicked_elements.append(element)
                else:
                    non_clicked_elements.append(element)

            # Step 4: Normalize vector sizes - ensure all vectors have the same number of parts
            # Find the maximum number of parts across all vectors
            max_length = max(
                [
                    elem.vector.shape[0]
                    for elem in clicked_elements + non_clicked_elements
                ]
            )

            # Pad clicked vectors to ensure uniform size
            for elem in clicked_elements:
                (
                    N,
                    D,
                ) = (
                    elem.vector.shape
                )  # N = number of parts, D = embedding dimension (typically 1024)

                if N < max_length:
                    # Pad shorter vectors with zeros to match max_length
                    pad_size = max_length - N
                    padding = torch.zeros(pad_size, D, dtype=elem.vector.dtype)
                    # After padding: Shape becomes [max_length, D]
                    elem.vector = torch.cat([elem.vector, padding], dim=0)

            # Pad non-clicked vectors similarly
            for elem in non_clicked_elements:
                (
                    N,
                    D,
                ) = (
                    elem.vector.shape
                )  # N = number of parts, D = embedding dimension

                if N < max_length:
                    # Pad shorter vectors with zeros
                    pad_size = max_length - N
                    padding = torch.zeros(pad_size, D, dtype=elem.vector.dtype)
                    # After padding: Shape becomes [max_length, D]
                    elem.vector = torch.cat([elem.vector, padding], dim=0)

            # Create improvement input for this session
            improvement_inputs.append(
                ImprovementInput(
                    session_id=session.session_id,
                    query=ImprovementElement(
                        id=query_object.object_id,
                        vector=query_vector,  # Shape: [N, D]
                        user_id=session.user_id,
                    ),
                    clicked_elements=clicked_elements,  # List of elements with vectors of shape [max_length, D]
                    non_clicked_elements=non_clicked_elements,  # List of elements with vectors of shape [max_length, D]
                )
            )

        # Step 5: Apply vector adjustment if we have valid inputs
        if len(improvement_inputs) > 0:
            # Get information about the vector space (distance metric, etc.)
            state_info = blue_collection.get_state_info()
            # Load the appropriate vector adjuster plugin
            plugin = plugin_manager.get_plugin(state_info.embedding_model.name)
            adjuster = plugin.get_vectors_adjuster()

            # Run the vector adjustment algorithm
            # This transforms vectors to increase query-to-clicked similarity
            # and decrease query-to-non-clicked similarity
            improved_inputs = adjuster.adjust_vectors(improvement_inputs)

            # Step 6: Prepare adjusted vectors for database insertion
            objects_to_upsert = []
            for input in improved_inputs:
                user_id = session_to_user[input.session_id]
                # Process both clicked and non-clicked elements
                for element in (
                    input.clicked_elements + input.non_clicked_elements
                ):
                    object_id = element.id
                    # Create personalized ID by appending user_id if not already personalized
                    new_object_id = (
                        object_id
                        if object_id in not_originals
                        else f"{object_id}_{user_id}"
                    )

                    # Convert tensor parts back to database-compatible format
                    object_parts = []
                    # Only use valid parts (not padding) by checking against is_average length
                    for i, vector in enumerate(
                        element.vector.tolist()[: len(element.is_average)]
                    ):
                        object_parts.append(
                            ObjectPart(
                                vector=vector,  # Individual part vector, shape: [D]
                                part_id=f"{new_object_id}:{'average' if element.is_average[i] else i}",
                                is_average=element.is_average[i],
                                user_id=user_id,
                            )
                        )

                    # Create complete object with improved vectors
                    objects_to_upsert.append(
                        Object(
                            object_id=new_object_id,  # Personalized ID
                            original_id=object_id,  # Original ID for reference
                            user_id=user_id,
                            session_id=object_by_id[object_id].session_id,
                            payload=object_by_id[object_id].payload,
                            storage_meta=object_by_id[object_id].storage_meta,
                            parts=object_parts,  # Improved vector parts
                        )
                    )

            # Insert or update the personalized vectors in the database
            if len(objects_to_upsert) > 0:
                blue_collection.upsert(objects_to_upsert)

    except Exception:
        # Handle any errors during the improvement process
        logger.exception("Something went wrong during vectors adjustment")
        for session_for_improvement in sessions_for_improvement:
            # Mark all sessions as failed
            session_for_improvement.status = TaskStatus.failed
            # Truncate error details if too long
            detail = traceback.format_exc()
            if len(detail) > 1500:
                detail = detail[-1500:]
            session_for_improvement.detail = detail
            context.sessions_for_improvement.update(
                obj=session_for_improvement
            )

        return

    # Success path - mark all sessions as complete
    logger.info("Vectors adjustment is finished successfully")
    for session_for_improvement in sessions_for_improvement:
        session_for_improvement.status = TaskStatus.done
        context.sessions_for_improvement.update(obj=session_for_improvement)
