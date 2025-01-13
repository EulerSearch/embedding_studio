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
    sessions = []
    for session in sessions_for_improvement:
        try:
            sessions.append(
                context.clickstream_dao.get_session(session.session_id)
            )

        except Exception:
            logger.exception(
                f"Something went wrong during retrieveing session with ID: {session.session_id}"
            )
            session.status = TaskStatus.failed
            session.detail = traceback.format_exc()
            context.sessions_for_improvement.update(obj=session)

    plugin_manager = PluginManager()
    # Initialize and discover plugins
    plugin_manager.discover_plugins(directory=settings.ES_PLUGINS_PATH)

    try:
        blue_collection = context.vectordb.get_blue_collection()
        blue_query_collection = context.vectordb.get_blue_query_collection()

    except Exception:
        logger.exception(
            f"Something went wrong during retrieveing blue collection"
        )
        for session in sessions_for_improvement:
            session.status = TaskStatus.failed
            session.detail = traceback.format_exc()
            context.sessions_for_improvement.update(obj=session)

        return

    try:
        # Prepare data for vectors adjustment
        improvement_inputs = []
        not_originals = set()
        object_to_originals = dict()
        object_by_id = dict()
        session_to_user = dict()

        for session in sessions:
            # Retrieve related query vectors
            session_to_user[session.session_id] = session.user_id
            query_object = blue_query_collection.get_objects_by_session_id(
                session.session_id
            )[0]
            query_vector = []
            for part in query_object.parts:
                query_vector.append(part.vector)
            query_vector = torch.Tensor(query_vector)

            # And distribute session items by clicked and non-clicked groups
            clicked_object_ids = set()
            if not session.is_irrelevant:
                for event in session.events:
                    clicked_object_ids.add(event.object_id)

            # Retrieve related item vectors
            clicked_elements = []
            non_clicked_elements = []

            for res in session.search_results:
                res_obj = blue_collection.find_by_ids(
                    [
                        res.object_id,
                    ]
                )[0]
                if res_obj.original_id is not None:
                    not_originals.add(res_obj.object_id)
                    object_to_originals[
                        res_obj.object_id
                    ] = res_obj.original_id

                object_by_id[res_obj.object_id] = res_obj

                res_vector = []
                for part in res_obj.parts:
                    res_vector.append(part.vector)
                res_vector = torch.Tensor(res_vector)
                element = ImprovementElement(
                    id=res_obj.object_id,
                    vector=res_vector,
                )

                if res.object_id in clicked_object_ids:
                    clicked_elements.append(element)

                else:
                    non_clicked_elements.append(element)

            improvement_inputs.append(
                ImprovementInput(
                    session_id=session.session_id,
                    query=ImprovementElement(
                        id=query_object.object_id, vector=query_vector
                    ),
                    clicked_elements=clicked_elements,
                    non_clicked_elements=non_clicked_elements,
                )
            )

        state_info = blue_collection.get_state_info()

        plugin = plugin_manager.get_plugin(state_info.embedding_model.name)
        adjuster = plugin.get_vectors_adjuster()

        # Process improvement
        improved_inputs = adjuster.adjust_vectors(improvement_inputs)

        # And prepare new vectors to upsert
        objects_to_upsert = []
        for input in improved_inputs:
            user_id = session_to_user[input.session_id]
            for element in input.clicked_elements + input.non_clicked_elements:
                object_id = element.id
                new_object_id = (
                    object_id
                    if object_id in not_originals
                    else f"{object_id}_{user_id}"
                )

                object_parts = []
                for i, vector in enumerate(element.vector.tolist()):
                    object_parts.append(
                        ObjectPart(
                            vector=vector, part_id=f"{new_object_id}:{i}"
                        )
                    )

                objects_to_upsert.append(
                    Object(
                        object_id=new_object_id,
                        original_id=object_by_id[object_id].original_id,
                        user_id=user_id,
                        session_id=object_by_id[object_id].session_id,
                        payload=object_by_id[object_id].payload,
                        storage_meta=object_by_id[object_id].storage_meta,
                        parts=object_parts,
                    )
                )

        if len(objects_to_upsert) > 0:
            blue_collection.upsert(objects_to_upsert)

    except Exception:
        logger.exception("Something went wrong during vectors adjustment")
        for session in sessions_for_improvement:
            session.status = TaskStatus.failed
            session.detail = traceback.format_exc()
            context.sessions_for_improvement.update(obj=session)

        return

    logger.info("Vectors adjustment is finished successfully")
    for session in sessions_for_improvement:
        session.status = TaskStatus.done
        context.sessions_for_improvement.update(obj=session)
