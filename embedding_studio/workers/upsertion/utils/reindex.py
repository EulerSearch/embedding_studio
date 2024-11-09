import logging
import time
import traceback
from typing import List, Optional

from dramatiq import Actor

from embedding_studio.context.app_context import context
from embedding_studio.core.config import settings
from embedding_studio.core.plugin import PluginManager
from embedding_studio.models.items_handler import FailedDataItem
from embedding_studio.models.reindex import (
    ReindexSubtaskCreateSchema,
    ReindexSubtaskInDb,
    ReindexTaskInDb,
)
from embedding_studio.models.task import TaskStatus
from embedding_studio.utils.dramatiq_task_handler import create_and_send_task
from embedding_studio.vectordb.collection import Collection

logger = logging.getLogger(__name__)

# Initialize plugin manager and discover plugins
plugin_manager = PluginManager()
plugin_manager.discover_plugins(directory=settings.ES_PLUGINS_PATH)


def process_reindex(
    task: ReindexTaskInDb, collection: Collection, reindex_subworker: Actor
):
    """
    Main loop for processing the reindex task.
    """
    total = collection.get_total()
    batch_size = settings.REINDEX_BATCH_SIZE
    batches_count = total // batch_size + 1

    offsets = []
    for i in range(batches_count):
        start = min(i * batch_size, total)
        end = min(start + batch_size, total)

        if start != end:
            offsets.append(start)

    max_tasks_count = settings.REINDEX_MAX_SUBTASKS_COUNT
    processing_task_ids = []

    while len(offsets) > 0:
        # Check and update status of processing tasks
        processing_task_ids = update_processing_tasks(
            task, processing_task_ids
        )

        # Create additional tasks if needed
        additional_tasks_count = max_tasks_count - len(processing_task_ids)
        if additional_tasks_count > 0:
            offsets = create_additional_tasks(
                task,
                offsets,
                batch_size,
                additional_tasks_count,
                processing_task_ids,
                reindex_subworker,
            )

        else:
            time.sleep(
                settings.REINDEX_WORKER_LOOP_WAIT_TIME
            )  # Wait before next iteration

    while len(processing_task_ids) > 0:
        for reindex_subtask_id in processing_task_ids.copy():
            reindex_subtask = context.reindex_subtask.get(reindex_subtask_id)

            if reindex_subtask.status not in [
                TaskStatus.processing,
                TaskStatus.pending,
            ]:
                # We risk getting stuck in an infinite loop here.
                # TODO: We may need to set some large timeout after which the task would be considered dead.
                processing_task_ids.remove(reindex_subtask_id)

        time.sleep(settings.REINDEX_WORKER_LOOP_WAIT_TIME)

    if settings.REINDEX_WORKER_MAX_FAILED != -1:
        failed_count = len(task.failed_items)
        if isinstance(settings.REINDEX_WORKER_MAX_FAILED, float):
            max_failed = int(settings.REINDEX_WORKER_MAX_FAILED * total)
            if failed_count >= max_failed:
                logger.error(
                    f"Task [{task.id}] is failed: failed count "
                    f"{failed_count} >= ({max_failed}={settings.REINDEX_WORKER_MAX_FAILED}x{total})"
                )
                task.status = TaskStatus.failed
                context.reindex_task.update(obj=task)
                return

        elif isinstance(settings.REINDEX_WORKER_MAX_FAILED, int):
            max_failed = settings.REINDEX_WORKER_MAX_FAILED
            if failed_count >= max_failed:
                logger.error(
                    f"Task [{task.id}] is failed: failed count {failed_count} >= {max_failed}"
                )
                task.status = TaskStatus.failed
                context.reindex_task.update(obj=task)
                return

    logger.error(
        f"Task [{task.id}] is done sucessfully: failed count {failed_count}"
    )
    # All done, update task status
    task.status = TaskStatus.done
    context.reindex_task.update(obj=task)


def update_processing_tasks(
    task: ReindexTaskInDb, processing_task_ids: List[str]
) -> List[str]:
    """
    Check the status of processing tasks and update the main task accordingly.
    """
    not_finished_processing_task_ids = []
    for reindex_subtask_id in processing_task_ids:
        reindex_subtask = context.reindex_subtask.get(reindex_subtask_id)

        if reindex_subtask.status in [
            TaskStatus.processing,
            TaskStatus.pending,
        ]:
            not_finished_processing_task_ids.append(reindex_subtask_id)
        elif reindex_subtask.status == TaskStatus.done:
            task.failed_items += reindex_subtask.failed_items
            task.add_count(len(reindex_subtask.items))
        elif reindex_subtask.status == TaskStatus.failed:
            task.failed_items += reindex_subtask.failed_items
        elif reindex_subtask.status == TaskStatus.refused:
            for item in reindex_subtask.items:
                failed_item = FailedDataItem.model_validate(item.dump())
                failed_item.detail = (
                    f"Upsertion task [{reindex_subtask.id}] is refused"
                )
                task.failed_items.append(failed_item)

    context.reindex_task.update(obj=task)
    return not_finished_processing_task_ids


def create_additional_tasks(
    task: ReindexTaskInDb,
    offsets: List[int],
    limit: int,
    additional_tasks_count: int,
    processing_task_ids: List[str],
    reindex_subworker: Actor,
) -> List[int]:
    """
    Create additional upsertion tasks as needed.
    Returns list of offsets, which weren't used.
    """
    offsets_to_keep = offsets[additional_tasks_count:]
    for i in range(additional_tasks_count):
        new_reindex_subtask = create_subtask(task, limit, offsets[i])

        updated_new_reindex_subtask = create_and_send_task(
            reindex_subworker, new_reindex_subtask, context.reindex_subtask
        )

        if not updated_new_reindex_subtask:
            handle_failed_subtask(task, new_reindex_subtask)
        else:
            processing_task_ids.append(updated_new_reindex_subtask.id)
            task.children.append(updated_new_reindex_subtask.id)

        context.reindex_task.update(obj=task)
        time.sleep(settings.REINDEX_WORKER_LOOP_WAIT_TIME)

    return offsets_to_keep


def create_subtask(
    task: ReindexTaskInDb, limit: int, offset: Optional[int] = None
) -> ReindexSubtaskInDb:
    """
    Create a new upsertion task for a batch of items.
    """
    return context.reindex_subtask.create(
        schema=ReindexSubtaskCreateSchema(
            source=task.source,
            dest=task.dest,
            limit=limit,
            offset=offset,
            parent_id=task.id,
        ),
        return_obj=True,
    )


def handle_failed_subtask(
    task: ReindexTaskInDb, failed_task: ReindexSubtaskInDb
):
    """
    Handle items from a failed upsertion task.
    """
    for item in failed_task.items:
        failed_item = FailedDataItem.model_validate(item.dump())
        failed_item.detail = "Unable to create upsertion task for item"
        task.failed_items.append(failed_item)


def handle_reindex_error(task: ReindexTaskInDb, error: Exception):
    """
    Handle errors that occur during the reindex process.
    """
    traceback_str = traceback.format_exc()
    logger.exception(f"Something went wrong while reindexing: {str(error)}")

    task.status = TaskStatus.failed
    task.detail = traceback_str[-1500:]
    context.reindex_task.update(obj=task)
