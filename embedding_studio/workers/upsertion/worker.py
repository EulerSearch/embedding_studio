import gc
import logging
import traceback

import dramatiq

from embedding_studio.context.app_context import context
from embedding_studio.core.config import settings
from embedding_studio.db.redis import redis_broker
from embedding_studio.models.delete import DeletionTaskCreateSchema
from embedding_studio.models.reindex_lock import ReindexLockCreateSchema
from embedding_studio.models.task import TaskStatus
from embedding_studio.models.upsert import UpsertionTaskCreateSchema
from embedding_studio.utils.dramatiq_middlewares import (
    ActionsOnStartMiddleware,
)
from embedding_studio.utils.dramatiq_task_handler import create_and_send_task
from embedding_studio.utils.initializer_actions import init_nltk
from embedding_studio.workers.inference.worker import (
    model_deletion_worker,
    model_deployment_worker,
)
from embedding_studio.workers.upsertion.handlers.delete import handle_delete
from embedding_studio.workers.upsertion.handlers.reindex import handle_reindex
from embedding_studio.workers.upsertion.handlers.reindex_subtask import (
    handle_reindex_subtask,
)
from embedding_studio.workers.upsertion.handlers.upsert import handle_upsert
from embedding_studio.workers.upsertion.utils.exceptions import (
    DeletionException,
    ReindexException,
    UpsertionException,
)

# Set up logging
logger = logging.getLogger(__name__)

redis_broker.add_middleware(ActionsOnStartMiddleware([init_nltk]))


@dramatiq.actor(
    queue_name="deletion_worker",
    max_retries=settings.DELETION_WORKER_MAX_RETRIES,
    time_limit=settings.DELETION_WORKER_TIME_LIMIT,
)
def deletion_worker(task_id: str):
    task = context.deletion_task.get(id=task_id)

    # TODO: mechanism to unlock if python service crashed
    reindex_lock = context.reindex_locks.get_by_model_id(
        task.embedding_model_id
    )
    if reindex_lock is not None:
        if settings.DELETION_PASS_TO_REINDEXING_MODEL:
            logger.warning(
                f"Passing deletion task to "
                f"reindexing model with ID[{reindex_lock.dst_embedding_model_id}]."
            )

            context.deletion_task.remove(task_id)

            iteration = context.mlflow_client.get_iteration_by_id(
                reindex_lock.dst_embedding_model.id
            )
            if iteration is None:
                task.status = TaskStatus.failed
                raise DeletionException(
                    f"Fine tuning iteration with ID"
                    f"[{reindex_lock.dst_embedding_model.id}] does not exist."
                )

            task = context.deletion_task.create(
                schema=DeletionTaskCreateSchema(
                    embedding_model_id=reindex_lock.dst_embedding_model.id,
                    object_ids=task.object_ids,
                ),
                return_obj=True,
                id=task.task_id,  # Use the provided task_id if available
            )

            # Use create_and_send_task instead of manual sending and updating
            updated_task = create_and_send_task(
                deletion_worker, task, context.deletion_task
            )

            if not updated_task:
                raise DeletionException(
                    f"Something went wrong while passing "
                    f"deletion task to reindexing model "
                    f"with ID[{reindex_lock.dst_embedding_model_id}]."
                )

            else:
                logger.info(
                    f"Task [{task_id}] is being passed to"
                    f" reindexing model with ID[{reindex_lock.dst_embedding_model_id}]."
                )

        else:
            logger.warning(
                f"Can't run deletion for embedding model {task.embedding_model_id}"
                f": it is locked while reindexing."
            )
            task.status = TaskStatus.refused
            context.deletion_task.update(obj=task)

        return

    handle_delete(task)


@dramatiq.actor(
    queue_name="upsertion_worker",
    max_retries=settings.UPSERTION_WORKER_MAX_RETRIES,
    time_limit=settings.UPSERTION_WORKER_TIME_LIMIT,
)
def upsertion_worker(task_id: str):
    task = context.upsertion_task.get(id=task_id)

    # TODO: mechanism to unlock if python service crashed
    reindex_lock = context.reindex_locks.get_by_model_id(
        task.embedding_model_id
    )
    if reindex_lock is not None:
        if settings.UPSERTION_PASS_TO_REINDEXING_MODEL:
            logger.warning(
                f"Passing upsertion task to "
                f"reindexing model with ID[{reindex_lock.dst_embedding_model_id}]."
            )

            context.upsertion_task.remove(task_id)

            task = context.upsertion_task.create(
                schema=UpsertionTaskCreateSchema(
                    embedding_model_id=reindex_lock.dst_embedding_model.id,
                    items=task.items,
                ),
                return_obj=True,
                id=task.task_id,  # Use the provided task_id if available
            )

            # Use create_and_send_task instead of manual sending and updating
            updated_task = create_and_send_task(
                upsertion_worker, task, context.upsertion_task
            )

            if not updated_task:
                raise UpsertionException(
                    f"Something went wrong while passing "
                    f"upsertion task to reindexing"
                    f" model with ID[{reindex_lock.dst_embedding_model_id}]."
                )

            else:
                logger.info(
                    f"Task [{task_id}] is being passed "
                    f"to reindexing model with ID[{reindex_lock.dst_embedding_model_id}]."
                )

        else:
            logger.warning(
                f"Can't run deletion for embedding model"
                f" {task.embedding_model_id}: it is locked while reindexing."
            )
            task.status = TaskStatus.refused
            context.upsertion_task.update(obj=task)

        return

    handle_upsert(task)


@dramatiq.actor(
    queue_name="reindex_subworker",
    max_retries=settings.REINDEX_SUBWORKER_MAX_RETRIES,
    time_limit=settings.REINDEX_SUBWORKER_TIME_LIMIT,
)
def reindex_subworker(task_id: str):
    task = context.reindex_subtask.get(id=task_id)
    handle_reindex_subtask(task)
    gc.collect()
    return


@dramatiq.actor(
    queue_name="reindex_worker",
    max_retries=settings.REINDEX_WORKER_MAX_RETRIES,
    time_limit=settings.REINDEX_WORKER_TIME_LIMIT,
)
def reindex_worker(task_id: str):
    task = context.reindex_task.get(id=task_id)

    reindex_lock = context.reindex_locks.get_by_model_id(
        task.source.embedding_model_id
    )
    if reindex_lock is not None:
        message = (
            f"Can't run reindexing process for source embedding "
            + f"model with ID[{task.source.embedding_model_id}]: "
            + f"this model is already being used as a source for reindexing."
        )

        if task.wait_on_conflict:
            logger.warning(message)
            reindex_worker.send_with_options(
                task_id=task_id, delay=settings.REINDEX_TASK_DELAY_TIME
            )
            return

        task.status = TaskStatus.refused
        context.reindex_task.update(obj=task)

        raise ReindexException(message)

    reindex_lock = context.reindex_locks.get_by_model_id(
        task.dest.embedding_model_id
    )
    if reindex_lock is not None:
        message = (
            f"Can't run reindexing process for destination embedding "
            + f"model with ID[{task.dest.embedding_model_id}]: "
            + f"this model is already being used as a source  for reindexing."
        )

        if task.wait_on_conflict:
            logger.warning(message)
            reindex_worker.send_with_options(
                task_id=task_id, delay=settings.REINDEX_TASK_DELAY_TIME
            )
            return

        task.status = TaskStatus.refused
        context.reindex_task.update(obj=task)

        raise ReindexException(message)

    reindex_lock = context.reindex_locks.get_by_model_id(
        task.source.embedding_model_id
    )
    if reindex_lock is not None:
        message = (
            f"Can't run reindexing process for source embedding "
            + f"model with ID[{task.source.embedding_model_id}]: "
            + f"this model is already being used as a destination for reindexing."
        )

        if task.wait_on_conflict:
            logger.warning(message)
            reindex_worker.send_with_options(
                task_id=task_id, delay=settings.REINDEX_TASK_DELAY_TIME
            )
            return

        task.status = TaskStatus.refused
        context.reindex_task.update(obj=task)

        raise ReindexException(message)

    reindex_lock = context.reindex_locks.get_by_dst_model_id(
        task.dest.embedding_model_id
    )
    if reindex_lock is not None:
        message = (
            f"Can't run reindexing process for destination embedding "
            + f"model with ID[{task.dest.embedding_model_id}]: "
            + f"this model is already being used as a destination  for reindexing."
        )

        if task.wait_on_conflict:
            logger.warning(message)
            reindex_worker.send_with_options(
                task_id=task_id, delay=settings.REINDEX_TASK_DELAY_TIME
            )
            return

        task.status = TaskStatus.refused
        context.reindex_task.update(obj=task)

        raise ReindexException(message)

    model_deletion_task = context.model_deletion_task.get_by_model_id(
        task.source.embedding_model_id
    )
    if model_deletion_task is not None:
        task.status = TaskStatus.refused
        context.reindex_task.update(obj=task)

        raise ReindexException(
            f"Can't run reindexing process for source embedding "
            f"model with ID[{task.source.embedding_model_id}]: "
            f"this model is already in the process of deletion."
        )

    model_deletion_task = context.model_deletion_task.get_by_model_id(
        task.dest.embedding_model_id
    )
    if model_deletion_task is not None:
        task.status = TaskStatus.refused
        context.reindex_task.update(obj=task)

        raise ReindexException(
            f"Can't run reindexing process for destination embedding "
            f"model with ID[{task.dest.embedding_model_id}]: "
            f"this model is already in the process of deletion."
        )

    statuses = ["pending", "processing"]
    counts = {status: 0 for status in statuses}

    for status in statuses:
        skip = 0
        while True:
            # Fetch a batch of matching tasks
            batch = context.reindex_task.get_by_filter(
                {"status": status}, skip=skip, limit=100
            )
            counts[status] += len(batch)

            # If fewer results than the limit, it’s the last page
            if len(batch) < 100:
                break

            skip += 100

    if sum(counts.values()) > settings.REINDEX_MAX_TASKS_COUNT:
        message = (
            f"Can't run reindexing process for destination embedding "
            + f"model with ID[{task.dest.embedding_model_id}]: "
            + f"tasks capacity ({sum(counts.values())}) has been exceeded (max: {settings.REINDEX_MAX_TASKS_COUNT})."
        )

        if task.wait_on_conflict:
            logger.warning(message)
            reindex_worker.send_with_options(
                task_id=task_id, delay=settings.REINDEX_TASK_DELAY_TIME
            )
            return

        task.status = TaskStatus.refused
        context.reindex_task.update(obj=task)

        raise ReindexException(message)

    lock = None
    try:
        lock = context.reindex_locks.create(
            schema=ReindexLockCreateSchema(
                embedding_model_id=task.source.embedding_model_id,
                dst_embedding_model_id=task.dest.embedding_model_id,
            ),
            id=task_id,
            return_obj=True,
        )

        if lock is None:
            task.status = TaskStatus.failed
            context.reindex_task.update(obj=task)

            raise ReindexException(
                f"Unable to create reindexing lock for task with ID[{task_id}]."
            )

        handle_reindex(
            task,
            reindex_subworker,
            deployment_worker=model_deployment_worker,
            deletion_worker=model_deletion_worker,
        )

    except Exception as e:
        traceback_str = traceback.format_exc()

        task.status = TaskStatus.failed
        task.detail = traceback_str[-1500:]
        context.reindex_task.update(obj=task)

        logger.exception(f"Something went wrong while reindexing: {str(e)}")

    finally:
        if lock is not None:
            # TODO: mechanism to unlock if python service crashed
            context.reindex_locks.remove(lock.id)

    return
